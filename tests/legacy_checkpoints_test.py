# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import json
import os
import tempfile

import torch
from torch.testing import assert_close

from lm_engine.hf_generation import HFGPTBaseForCausalLM
from lm_engine.legacy_checkpoints import CONFIG_BACKFILL
from lm_engine.modeling_utils.mlp_blocks.mlp.utils import split_up_gate_tensor_for_mlp
from lm_engine.models import GPTBaseConfig
from lm_engine.utils import SafeTensorsWeightsManager

from .utils import get_hybrid_m2rnn_test_config, get_m2rnn_moe_test_config


def _deinterleave(tensor, dim: int):
    u, g = split_up_gate_tensor_for_mlp(tensor, dim=dim)
    return torch.cat([u, g], dim=dim)


def _save_as_legacy_checkpoint(config_dict: dict, state_dict: dict, save_directory: str) -> None:
    for key in CONFIG_BACKFILL:
        del config_dict[key]
    with open(os.path.join(save_directory, "config.json"), "w") as f:
        json.dump(config_dict, f)
    SafeTensorsWeightsManager.save_state_dict(state_dict, save_directory)


def test_from_pretrained_loads_legacy_checkpoint() -> None:
    """Checkpoints exported before #465/#478 (like all open-lm-engine hub checkpoints) lack the
    new config fields and store GLU weights as [up; gate] halves instead of interleaved.
    from_pretrained must backfill the config and re-interleave the weights."""

    torch.manual_seed(42)
    config_dict = get_hybrid_m2rnn_test_config().to_dict()
    config_dict["tie_word_embeddings"] = True
    for mlp_block in config_dict["mlp_blocks"]:
        mlp_block["activation_function"] = "swiglu"  # the released checkpoints use GLU MLPs
    model = HFGPTBaseForCausalLM(GPTBaseConfig(**config_dict))
    model.eval()

    # re-seed so the inputs don't depend on how many RNG draws model construction consumed
    torch.manual_seed(0)
    input_ids = torch.randint(3, model.config.vocab_size, (2, 8))
    with torch.no_grad():
        expected_logits = model(input_ids=input_ids).logits

    state_dict = model.state_dict()
    for name in list(state_dict):
        if name.endswith("mlp_block.c_fc.weight"):
            state_dict[name] = _deinterleave(state_dict[name], dim=0)

    with tempfile.TemporaryDirectory() as save_directory:
        _save_as_legacy_checkpoint(config_dict, state_dict, save_directory)
        loaded_model = HFGPTBaseForCausalLM.from_pretrained(save_directory)

    loaded_model.eval()
    with torch.no_grad():
        loaded_logits = loaded_model(input_ids=input_ids).logits

    assert_close(loaded_logits, expected_logits)


def test_from_pretrained_loads_legacy_moe_checkpoint() -> None:
    """The 7B MoE checkpoints carry pre-#465 per-block interleave flags: routed experts were
    exported interleaved (use_interleaved_weights: true) while shared experts were not (flag
    absent = false). from_pretrained must pop the removed flags and convert exactly the
    tensors they mark as concatenated."""

    torch.manual_seed(42)
    config_dict = get_m2rnn_moe_test_config().to_dict()
    model = HFGPTBaseForCausalLM(GPTBaseConfig(**config_dict))
    model.eval()

    torch.manual_seed(0)
    input_ids = torch.randint(3, model.config.vocab_size, (2, 8))
    with torch.no_grad():
        expected_logits = model(input_ids=input_ids).logits

    # block 0 mimics the 7B checkpoints (routed already interleaved); the rest are fully
    # legacy — covers both flag values and both tensor layouts (3D routed dim=1, 2D shared dim=0)
    state_dict = model.state_dict()
    for name in list(state_dict):
        if name.endswith("mlp_block.c_fc.weight") and not name.startswith("transformer.h.0."):
            state_dict[name] = _deinterleave(state_dict[name], dim=1)
        elif name.endswith("mlp_block.c_fc_shared.weight"):
            state_dict[name] = _deinterleave(state_dict[name], dim=0)

    for i, mlp_block in enumerate(config_dict["mlp_blocks"]):
        mlp_block["use_interleaved_weights"] = i == 0
        # use_interleaved_weights_for_shared_experts stays absent, like the released checkpoints

    with tempfile.TemporaryDirectory() as save_directory:
        _save_as_legacy_checkpoint(config_dict, state_dict, save_directory)
        loaded_model = HFGPTBaseForCausalLM.from_pretrained(save_directory)

    loaded_model.eval()
    with torch.no_grad():
        loaded_logits = loaded_model(input_ids=input_ids).logits

    assert_close(loaded_logits, expected_logits)
