# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import json
import os

import torch
from transformers import GenerationConfig
from transformers.generation import GenerationMixin

from .modeling_utils.mlp_blocks.mlp.module import MLP
from .modeling_utils.mlp_blocks.mlp.utils import interleave_up_gate_tensor_for_mlp
from .modeling_utils.mlp_blocks.moe.module import MoE
from .modeling_utils.softplus_decay_gate import SoftplusDecayGate
from .models import GPTBaseForCausalLM


# Fields required by the current config schema but absent from configs exported before the
# config refactor (#478). The init-method fields only affect weight initialization; all
# released open-lm-engine checkpoints have tied embeddings (no lm_head.weight on the hub).
_CONFIG_BACKFILL = {
    "embedding_init_method": "normal",
    "use_depth_scaled_init": False,
    "tie_word_embeddings": True,
}

# per-mlp-block config keys that existed before #465; their values say which fused GLU
# tensors are already interleaved in the checkpoint (absent = False = concatenated [up; gate])
_LEGACY_INTERLEAVE_KEYS = ("use_interleaved_weights", "use_interleaved_weights_for_shared_experts")


def _interleave(tensor: torch.Tensor, dim: int) -> None:
    u, g = tensor.chunk(2, dim=dim)
    tensor.copy_(interleave_up_gate_tensor_for_mlp(u, g, dim=dim))


@torch.no_grad()
def _restore_declared_dtypes(model: GPTBaseForCausalLM) -> None:
    """SoftplusDecayGate declares A_log/dt_bias as fp32 (their values live in softplus/exp
    ranges where bf16 loses real precision), but from_pretrained's blanket model.to(dtype)
    downcasts them. Restore fp32 storage; the module's forward already upcasts internally,
    so mixed dtypes are safe in every path including no-autocast generation."""

    for module in model.modules():
        if isinstance(module, SoftplusDecayGate):
            module.A_log.data = module.A_log.data.float()
            module.dt_bias.data = module.dt_bias.data.float()


@torch.no_grad()
def _interleave_legacy_glu_weights(model: GPTBaseForCausalLM, interleave_flags: list[tuple[bool, bool]]) -> None:
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

        if isinstance(mlp, MLP):
            if not experts_interleaved:
                _interleave(mlp.c_fc.weight, dim=0)
                if mlp.c_fc.bias is not None:
                    _interleave(mlp.c_fc.bias, dim=0)
        elif isinstance(mlp, MoE):
            if not experts_interleaved:
                # routed experts: (num_experts, 2 * intermediate, hidden)
                _interleave(mlp.c_fc.weight, dim=1)
                if mlp.c_fc.bias is not None:
                    _interleave(mlp.c_fc.bias, dim=1)
            if getattr(mlp, "c_fc_shared", None) is not None and not shared_interleaved:
                _interleave(mlp.c_fc_shared.weight, dim=0)
                if mlp.c_fc_shared.bias is not None:
                    _interleave(mlp.c_fc_shared.bias, dim=0)
        else:
            raise NotImplementedError(f"unknown mlp block type {type(mlp).__name__}")

# kwargs passed by transformers/TRL that have no lm-engine equivalent and can be dropped
# without changing forward's semantics
_IGNORABLE_FORWARD_KWARGS = {"cache_position", "output_attentions", "output_hidden_states", "return_dict"}


class HFGPTBaseForCausalLM(GenerationMixin, GPTBaseForCausalLM):
    """GPTBaseForCausalLM that speaks transformers' GenerationMixin protocol.

    The recurrent/hybrid cache (`cache_params`) is threaded through the HF generate loop the
    same way transformers handles Mamba: the model is marked stateful so that generate()
    creates no `past_key_values`, and `prepare_inputs_for_generation` /
    `_update_model_kwargs_for_generation` carry `cache_params` between steps instead.
    """

    main_input_name = "input_ids"
    _is_stateful = True

    def __init__(self, config, **kwargs) -> HFGPTBaseForCausalLM:
        super().__init__(config, **kwargs)
        self.generation_config = GenerationConfig.from_model_config(config)

    def add_model_tags(self, tags) -> None:
        """hub-metadata no-op (PreTrainedModel API expected by TRL)"""

    @property
    def is_gradient_checkpointing(self) -> bool:
        # PreTrainedModel API expected by TRL; mirrors its definition
        return any(getattr(module, "gradient_checkpointing", False) for module in self.modules())

    # GenerationMixin expects these; nn.Module doesn't provide them
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> HFGPTBaseForCausalLM:
        if not os.path.isdir(pretrained_model_name_or_path):
            from huggingface_hub import snapshot_download

            pretrained_model_name_or_path = snapshot_download(
                pretrained_model_name_or_path, allow_patterns=["*.json", "*.safetensors", "tokenizer*"]
            )

        config_dict = json.load(open(os.path.join(pretrained_model_name_or_path, "config.json")))
        # missing schema fields mean the checkpoint was exported before the config refactor
        # (#478) and therefore also before the interleaved-weights breaking change (#465)
        is_legacy_checkpoint = any(key not in config_dict for key in _CONFIG_BACKFILL)

        # pre-#465 per-block interleave flags no longer exist in the schema: pop them (the
        # pydantic config forbids unknown fields) and let them drive the weight conversion
        interleave_flags = [
            tuple(mlp_block.pop(key, False) for key in _LEGACY_INTERLEAVE_KEYS)
            for mlp_block in config_dict.get("mlp_blocks", [])
        ]

        if "config" not in kwargs:
            for key, value in _CONFIG_BACKFILL.items():
                config_dict.setdefault(key, value)

            kwargs["config"] = cls.config_class.from_dict(config_dict)

        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        if is_legacy_checkpoint:
            _interleave_legacy_glu_weights(model, interleave_flags)

        _restore_declared_dtypes(model)

        return model

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cache_params=None,
        past_key_values=None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ):
        assert past_key_values is None, "use cache_params instead of past_key_values"

        unknown_kwargs = set(kwargs) - _IGNORABLE_FORWARD_KWARGS
        assert len(unknown_kwargs) == 0, f"forward got unexpected kwargs: {unknown_kwargs}"

        return super().forward(
            input_ids=input_ids,
            cache_params=cache_params,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            logits_to_keep=0 if logits_to_keep is None else logits_to_keep,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        use_cache: bool | None = None,
        cache_params=None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        # prefill: cache_params is None and the model creates the cache when use_cache is set
        # decode: the state already covers everything but the last token
        if cache_params is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "cache_params": cache_params,
            "use_cache": use_cache,
            # sampling only reads the last position; saves prompt_length x vocab_size logits
            # during prefill
            "logits_to_keep": 1,
        }

    def _update_model_kwargs_for_generation(
        self, outputs, model_kwargs: dict, num_new_tokens: int = 1, **kwargs
    ) -> dict:
        model_kwargs["cache_params"] = getattr(outputs, "cache_params", None)

        if model_kwargs.get("attention_mask") is not None:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.size(0), 1))], dim=-1
            )

        return model_kwargs
