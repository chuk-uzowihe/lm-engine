# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

"""Backwards compatibility for checkpoints exported before the config refactor (#478) and the
interleaved-GLU-weights breaking change (#465) — which includes every released open-lm-engine
hub checkpoint. HFGPTBaseForCausalLM.from_pretrained drives these at load time; nothing here
runs for checkpoints exported by current code."""

from __future__ import annotations

import torch

from .modeling_utils.mlp_blocks.mlp.utils import interleave_up_gate_tensor_for_mlp
from .models import GPTBaseForCausalLM


# Fields required by the current config schema but absent from configs exported before the
# config refactor (#478). The init-method fields only affect weight initialization; all
# released open-lm-engine checkpoints have tied embeddings (no lm_head.weight on the hub).
CONFIG_BACKFILL = {
    "embedding_init_method": "normal",
    "use_depth_scaled_init": False,
    "tie_word_embeddings": True,
}

# per-mlp-block config keys that existed before #465; their values say which fused GLU
# tensors are already interleaved in the checkpoint (absent = False = concatenated [up; gate])
_LEGACY_INTERLEAVE_KEYS = ("use_interleaved_weights", "use_interleaved_weights_for_shared_experts")


def is_legacy_checkpoint(config_dict: dict) -> bool:
    """missing schema fields mean the checkpoint was exported before the config refactor
    (#478) and therefore also before the interleaved-weights breaking change (#465)"""
    return any(key not in config_dict for key in CONFIG_BACKFILL)


def backfill_config(config_dict: dict) -> None:
    """fill schema fields the legacy export predates (in place)"""
    for key, value in CONFIG_BACKFILL.items():
        config_dict.setdefault(key, value)


def pop_legacy_interleave_flags(config_dict: dict) -> list[tuple[bool, bool]]:
    """Remove the pre-#465 per-block interleave flags from the config dict (the pydantic
    config forbids unknown fields) and return them to drive the weight conversion."""
    return [
        tuple(mlp_block.pop(key, False) for key in _LEGACY_INTERLEAVE_KEYS)
        for mlp_block in config_dict.get("mlp_blocks", [])
    ]


def _interleave(tensor: torch.Tensor, dim: int) -> None:
    u, g = tensor.chunk(2, dim=dim)
    tensor.copy_(interleave_up_gate_tensor_for_mlp(u, g, dim=dim))


@torch.no_grad()
def interleave_legacy_glu_weights(model: GPTBaseForCausalLM, interleave_flags: list[tuple[bool, bool]]) -> None:
    """Convert GLU weights from checkpoints that predate #465 (support only interleaved_weights).

    Legacy fused c_fc tensors store [up; gate] as concatenated halves along the output-feature
    axis; current code reads even units as gate and odd as up. Per-block config flags (popped
    from the legacy config) say which tensors were already exported interleaved — the released
    7B MoE checkpoints have interleaved routed experts but concatenated shared experts.
    """

    for block, (experts_interleaved, shared_interleaved) in zip(
        model.transformer.h.values(), interleave_flags, strict=True
    ):
        mlp = block.mlp_block

        if not getattr(mlp, "is_glu", False):
            continue

        # MoE routed-expert tensors are 3D (num_experts, 2 * intermediate, hidden) with the
        # fused-GLU axis at dim 1; MLP and shared-expert tensors carry it at dim 0. Biases
        # have one fewer dim, so weight.ndim - 2 addresses the same axis in both.
        for linear, interleaved in (
            (mlp.c_fc, experts_interleaved),
            (getattr(mlp, "c_fc_shared", None), shared_interleaved),
        ):
            if linear is None or interleaved:
                continue
            dim = linear.weight.ndim - 2
            _interleave(linear.weight, dim=dim)
            if linear.bias is not None:
                _interleave(linear.bias, dim=dim)
