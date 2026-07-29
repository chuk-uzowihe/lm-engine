# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import json
import os
import tempfile

import pytest
import torch

from lm_engine.hf_generation import _CONFIG_BACKFILL, HFGPTBaseForCausalLM
from lm_engine.models import GPTBaseConfig, GPTBaseForCausalLM
from lm_engine.modeling_utils.mlp_blocks.mlp.utils import split_up_gate_tensor_for_mlp
from lm_engine.utils import SafeTensorsWeightsManager


def _deinterleave(tensor, dim: int):
    u, g = split_up_gate_tensor_for_mlp(tensor, dim=dim)
    return torch.cat([u, g], dim=dim)

from .utils import get_dense_test_config, skip_test_if_device_unavailable


def get_hybrid_m2rnn_test_config(num_layers: int = 4) -> GPTBaseConfig:
    """dense test config with all but one layer swapped from attention to m2rnn"""

    config_dict = get_dense_test_config("nope", num_layers=num_layers).to_dict()

    m2rnn_block = {
        "sequence_mixer_type": "m2rnn",
        "k_head_dim": 8,
        "v_head_dim": 8,
        "num_q_heads": 1,
        "num_k_heads": 1,
        "num_v_heads": 4,
        "num_f_heads": 4,
        "num_g_heads": 4,
        "num_weight_heads": 4,
        "use_residual": True,
        "kernel_size": 4,
        "activation_function": "silu",
        "add_bias": False,
        "gradient_clipping": 1.0,
        "normalization_function": "rmsnorm",
        "A_init_min": 0,
        "A_init_max": 16,
        "dt_init_min": 0.001,
        "dt_init_max": 0.1,
        "dt_init_floor": 0.0001,
    }

    config_dict["sequence_mixer_blocks"] = [
        config_dict["sequence_mixer_blocks"][i] if i == 1 else m2rnn_block for i in range(num_layers)
    ]

    return GPTBaseConfig(**config_dict)


def _get_model_and_inputs(
    device: torch.device, left_pad: int = 0
) -> tuple[HFGPTBaseForCausalLM, torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)

    config = get_hybrid_m2rnn_test_config()
    model = HFGPTBaseForCausalLM(config).to(device)
    model.eval()

    torch.manual_seed(0)
    # keep clear of bos/eos/pad (0/1/2) so eos doesn't fire inside the prompt
    input_ids = torch.randint(3, config.vocab_size, (2, 8), device=device)
    attention_mask = torch.ones_like(input_ids)

    if left_pad > 0:
        input_ids[0, :left_pad] = config.pad_token_id
        attention_mask[0, :left_pad] = 0

    return model, input_ids, attention_mask


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
def test_incremental_decode_matches_full_forward(device: torch.device) -> None:
    skip_test_if_device_unavailable(device)

    model, input_ids, _ = _get_model_and_inputs(device)

    with torch.no_grad():
        full_logits = model(input_ids=input_ids).logits

        output = model(input_ids=input_ids[:, :4], use_cache=True)
        step_logits = [output.logits]
        for t in range(4, input_ids.size(1)):
            output = model(input_ids=input_ids[:, t : t + 1], cache_params=output.cache_params, use_cache=True)
            step_logits.append(output.logits)

    torch.testing.assert_close(torch.cat(step_logits, dim=1), full_logits, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
@pytest.mark.parametrize("left_pad", [0, 3])
def test_hf_greedy_generate_matches_native(device: torch.device, left_pad: int) -> None:
    skip_test_if_device_unavailable(device)

    model, input_ids, attention_mask = _get_model_and_inputs(device, left_pad=left_pad)

    hf_output = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=16, do_sample=False
    )
    native_output = GPTBaseForCausalLM.generate(
        model, input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=16, temperature=0
    )

    torch.testing.assert_close(hf_output, native_output)


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
def test_hf_sampled_generate_matches_native(device: torch.device) -> None:
    skip_test_if_device_unavailable(device)

    model, input_ids, attention_mask = _get_model_and_inputs(device)

    torch.manual_seed(1234)
    hf_output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=16,
        do_sample=True,
        temperature=0.8,
        top_k=5,
    )

    torch.manual_seed(1234)
    native_output = GPTBaseForCausalLM.generate(
        model, input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=16, temperature=0.8, top_k=5
    )

    torch.testing.assert_close(hf_output, native_output)


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

    torch.manual_seed(0)
    input_ids = torch.randint(3, model.config.vocab_size, (2, 8))
    with torch.no_grad():
        expected_logits = model(input_ids=input_ids).logits

    state_dict = model.state_dict()
    for name in list(state_dict):
        if name.endswith("mlp_block.c_fc.weight"):
            u, g = split_up_gate_tensor_for_mlp(state_dict[name], dim=0)
            state_dict[name] = torch.cat([u, g], dim=0)

    for key in _CONFIG_BACKFILL:
        del config_dict[key]

    with tempfile.TemporaryDirectory() as save_directory:
        json.dump(config_dict, open(os.path.join(save_directory, "config.json"), "w"))
        SafeTensorsWeightsManager.save_state_dict(state_dict, save_directory)

        loaded_model = HFGPTBaseForCausalLM.from_pretrained(save_directory)

    loaded_model.eval()
    with torch.no_grad():
        loaded_logits = loaded_model(input_ids=input_ids).logits

    torch.testing.assert_close(loaded_logits, expected_logits)


def test_from_pretrained_keeps_decay_gate_params_fp32() -> None:
    """SoftplusDecayGate declares A_log/dt_bias fp32; loading in bf16 must not downcast them
    (the blanket model.to(dtype) in from_pretrained otherwise does)."""

    torch.manual_seed(42)
    config_dict = get_hybrid_m2rnn_test_config().to_dict()
    config_dict["tie_word_embeddings"] = True
    model = HFGPTBaseForCausalLM(GPTBaseConfig(**config_dict))

    with tempfile.TemporaryDirectory() as save_directory:
        model.save_pretrained(save_directory)
        loaded_model = HFGPTBaseForCausalLM.from_pretrained(save_directory, dtype=torch.bfloat16)

    assert loaded_model.transformer.wte.weight.dtype == torch.bfloat16
    for name, param in loaded_model.named_parameters():
        if "decay_gate" in name:
            assert param.dtype == torch.float32, name

    # mixed dtypes must survive a no-autocast forward (the generation path)
    loaded_model.eval()
    with torch.no_grad():
        logits = loaded_model(input_ids=torch.randint(3, 128, (1, 8))).logits
    assert torch.isfinite(logits.float()).all()


def test_from_pretrained_loads_legacy_moe_checkpoint() -> None:
    """The 7B MoE checkpoints carry pre-#465 per-block interleave flags: routed experts were
    exported interleaved (use_interleaved_weights: true) while shared experts were not (flag
    absent = false). from_pretrained must pop the removed flags and convert exactly the
    tensors they mark as concatenated."""

    torch.manual_seed(42)
    config_dict = get_hybrid_m2rnn_test_config().to_dict()
    config_dict["tie_word_embeddings"] = True
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

    for key in _CONFIG_BACKFILL:
        del config_dict[key]
    for i, mlp_block in enumerate(config_dict["mlp_blocks"]):
        mlp_block["use_interleaved_weights"] = i == 0
        # use_interleaved_weights_for_shared_experts stays absent, like the released checkpoints

    with tempfile.TemporaryDirectory() as save_directory:
        json.dump(config_dict, open(os.path.join(save_directory, "config.json"), "w"))
        SafeTensorsWeightsManager.save_state_dict(state_dict, save_directory)

        loaded_model = HFGPTBaseForCausalLM.from_pretrained(save_directory)

    loaded_model.eval()
    with torch.no_grad():
        loaded_logits = loaded_model(input_ids=input_ids).logits

    torch.testing.assert_close(loaded_logits, expected_logits)


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
def test_logits_to_keep(device: torch.device) -> None:
    skip_test_if_device_unavailable(device)

    model, input_ids, attention_mask = _get_model_and_inputs(device)

    with torch.no_grad():
        full_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        sliced_logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=3).logits

    torch.testing.assert_close(sliced_logits, full_logits[:, -3:])
