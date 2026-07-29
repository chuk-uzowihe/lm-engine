# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from lm_engine.hf_generation import HFGPTBaseForCausalLM
from lm_engine.models import GPTBaseConfig
from lm_engine.routing_replay import record_routing, replay_routing

from .hf_generation_test import get_hybrid_m2rnn_test_config


def _tiny_moe_model() -> HFGPTBaseForCausalLM:
    torch.manual_seed(42)
    config_dict = get_hybrid_m2rnn_test_config().to_dict()
    config_dict["router_aux_loss_coef"] = 0.001
    config_dict["mlp_blocks"] = [
        {
            "mlp_type": "MoE",
            "activation_function": "swiglu",
            "add_bias": False,
            "intermediate_size": 16,
            "shared_intermediate_size": 32,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "normalized_topk": True,
            "shared_expert_gating": False,
            "dropout": 0,
        }
        for _ in range(len(config_dict["sequence_mixer_blocks"]))
    ]
    model = HFGPTBaseForCausalLM(GPTBaseConfig(**config_dict))
    model.eval()
    return model


def _assemble(recorder, rows: int, seq_len: int) -> torch.Tensor:
    layers = [recorder.calls[tag][0].view(rows, seq_len, -1) for tag in sorted(recorder.calls)]
    return torch.stack(layers, dim=1)


def test_replayed_routing_reproduces_recorded_forward() -> None:
    """With normalized_topk, replaying a forward's own selection through the restricted
    softmax must reproduce it exactly — the invariant that makes R3 a no-op on-policy."""

    model = _tiny_moe_model()
    torch.manual_seed(0)
    input_ids = torch.randint(3, model.config.vocab_size, (2, 8))

    with torch.no_grad():
        with record_routing() as recorder:
            recorded_logits = model(input_ids=input_ids).logits

        num_moe_layers = len(recorder.calls)
        assert num_moe_layers == len(model.config.mlp_blocks)
        replay = _assemble(recorder, rows=2, seq_len=8)

        with replay_routing(replay):
            replayed_logits = model(input_ids=input_ids).logits

    torch.testing.assert_close(replayed_logits, recorded_logits)


def test_replayed_routing_overrides_selection() -> None:
    """A perturbed replay must actually change the forward (the override is real)."""

    model = _tiny_moe_model()
    torch.manual_seed(0)
    input_ids = torch.randint(3, model.config.vocab_size, (2, 8))

    with torch.no_grad():
        with record_routing() as recorder:
            recorded_logits = model(input_ids=input_ids).logits

        replay = _assemble(recorder, rows=2, seq_len=8)
        perturbed = (replay + 1) % 4  # shift every expert selection

        with replay_routing(perturbed):
            perturbed_logits = model(input_ids=input_ids).logits

    assert not torch.allclose(perturbed_logits, recorded_logits)


def test_replay_cursor_handles_chunked_forwards() -> None:
    """Replay slices rows per call, so chunked (micro-batched) forwards stay aligned."""

    model = _tiny_moe_model()
    torch.manual_seed(0)
    input_ids = torch.randint(3, model.config.vocab_size, (4, 8))

    with torch.no_grad():
        with record_routing() as recorder:
            recorded_logits = model(input_ids=input_ids).logits

        replay = _assemble(recorder, rows=4, seq_len=8)

        with replay_routing(replay):
            chunk_logits = torch.cat(
                [model(input_ids=input_ids[:2]).logits, model(input_ids=input_ids[2:]).logits]
            )

    torch.testing.assert_close(chunk_logits, recorded_logits)
