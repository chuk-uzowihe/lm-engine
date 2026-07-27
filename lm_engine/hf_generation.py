# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import json
import os

import torch
from transformers import GenerationConfig
from transformers.generation import GenerationMixin

from .models import GPTBaseForCausalLM


# Fields required by the current config schema but absent from configs exported before the
# config refactor (#478). The init-method fields only affect weight initialization; all
# released open-lm-engine checkpoints have tied embeddings (no lm_head.weight on the hub).
_CONFIG_BACKFILL = {
    "embedding_init_method": "normal",
    "use_depth_scaled_init": False,
    "tie_word_embeddings": True,
}

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

        if "config" not in kwargs:
            config_dict = json.load(open(os.path.join(pretrained_model_name_or_path, "config.json")))
            for key, value in _CONFIG_BACKFILL.items():
                config_dict.setdefault(key, value)

            kwargs["config"] = cls.config_class.from_dict(config_dict)

        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

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

        output = super().forward(
            input_ids=input_ids,
            cache_params=cache_params,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
        )

        # TRL and friends pass logits_to_keep to avoid materializing prompt logits; we slice
        # after the fact, which keeps semantics (memory optimization can come later)
        if isinstance(logits_to_keep, int):
            if logits_to_keep > 0 and output.logits is not None:
                output.logits = output.logits[:, -logits_to_keep:]
        elif logits_to_keep is not None:
            output.logits = output.logits[:, logits_to_keep]

        return output

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
