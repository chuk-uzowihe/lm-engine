# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

"""Rollout Routing Replay (R3, arxiv.org/abs/2510.11370) for RL on MoE models.

MoE routers are discontinuous: tiny numeric differences between the rollout forward and the
training forward select different experts for the same token, which destabilizes RL. R3
records which experts the rollout selected and replays that selection during training,
computing the gate weights as a softmax of the *training* router logits restricted to the
replayed expert set.

Two context managers, both consumed by MoE._compute_routing_weights via observe():

- record_routing(): capture mode. Stores each MoE layer's selected experts per forward call
  (layers are tagged in first-execution order, which is deterministic).
- replay_routing(replay): serve mode. `replay` is (num_rows, num_moe_layers, seq_len, top_k);
  each forward call consumes the next slice of rows for its layer, so chunked/micro-batched
  forwards stay aligned as long as row order is preserved.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F


_ACTIVE: "_RecordContext | _ReplayContext | None" = None


def get_active_routing_replay() -> "_RecordContext | _ReplayContext | None":
    return _ACTIVE


class _RecordContext:
    def __init__(self) -> None:
        self.layer_tags: dict[int, int] = {}
        # per layer tag: list of (num_tokens, top_k) index tensors, one per forward call
        self.calls: dict[int, list[torch.Tensor]] = {}

    def _tag(self, module) -> int:
        tag = self.layer_tags.setdefault(id(module), len(self.layer_tags))
        self.calls.setdefault(tag, [])
        return tag

    def observe(self, module, router_logits: torch.Tensor, selected_experts: torch.Tensor) -> None:
        assert module.num_experts <= torch.iinfo(torch.uint8).max + 1
        self.calls[self._tag(module)].append(selected_experts.to(torch.uint8))
        return None


class _ReplayContext:
    def __init__(self, replay: torch.Tensor) -> None:
        # (num_rows, num_moe_layers, seq_len, top_k)
        self.replay = replay
        self.layer_tags: dict[int, int] = {}
        self.row_cursor: dict[int, int] = {}

    def observe(self, module, router_logits: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
        tag = self.layer_tags.setdefault(id(module), len(self.layer_tags))

        num_rows, num_layers, seq_len, top_k = self.replay.shape
        assert tag < num_layers, "forward hit more MoE layers than the replay tensor holds"

        num_tokens = selected_experts.size(0)
        assert num_tokens % seq_len == 0, f"call of {num_tokens} tokens is not a multiple of seq_len {seq_len}"
        call_rows = num_tokens // seq_len

        start = self.row_cursor.get(tag, 0)
        assert start + call_rows <= num_rows, "replay rows exhausted; row order/chunking mismatch"
        self.row_cursor[tag] = start + call_rows

        replayed = self.replay[start : start + call_rows, tag]  # (call_rows, seq_len, top_k)
        return replayed.reshape(num_tokens, top_k).long()


@contextmanager
def record_routing():
    global _ACTIVE
    assert _ACTIVE is None, "routing replay contexts do not nest"
    context = _RecordContext()
    _ACTIVE = context
    try:
        yield context
    finally:
        _ACTIVE = None


@contextmanager
def replay_routing(replay: torch.Tensor):
    global _ACTIVE
    assert _ACTIVE is None, "routing replay contexts do not nest"
    context = _ReplayContext(replay)
    _ACTIVE = context
    try:
        yield context
    finally:
        _ACTIVE = None


def restricted_router_weights(router_logits: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
    """R3 gate weights: softmax of the (current) router logits over the replayed expert set,
    g_i = I_i * exp(s_i) / sum_j I_j * exp(s_j)"""
    return F.softmax(router_logits.gather(1, selected_experts).float(), dim=-1)
