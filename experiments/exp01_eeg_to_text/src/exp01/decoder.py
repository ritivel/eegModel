"""Gemma 4 decoder loader with frozen / LoRA modes and vocab extension."""

from __future__ import annotations

import torch
import torch.nn as nn

from . import storage


class LoadedDecoder(nn.Module):
    """Wraps a Gemma 4 IT model + tokenizer.

    Subclassing ``nn.Module`` matters because the tokenizer is held as a
    plain attribute (it isn't a Module) while ``self.model`` is registered
    as a submodule — so ``EEG2Text(...).to("cuda")`` recursively moves
    Gemma's weights too.
    """

    def __init__(self, model_id: str, *, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=str(storage.HF_CACHE), padding_side="left",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=str(storage.HF_CACHE),
            torch_dtype=dtype,
            attn_implementation="sdpa",
        )
        embed = self.model.get_input_embeddings()
        self.embed_dim = embed.embedding_dim
        self.vocab_size = embed.num_embeddings


def load_decoder(model_id: str = "google/gemma-4-E2B-it", *, dtype: torch.dtype = torch.bfloat16) -> LoadedDecoder:
    return LoadedDecoder(model_id, dtype=dtype)


def freeze(model: nn.Module) -> nn.Module:
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


def attach_lora(model: nn.Module, *, r: int = 16, alpha: int = 32, dropout: float = 0.05) -> nn.Module:
    """Wrap with PEFT LoRA on the language-model attention projections.

    Gemma 4 E2B is multimodal: ``model.model`` has ``vision_tower``,
    ``audio_tower``, and ``language_model`` children. The vision/audio
    towers wrap their q/k/v/o ``nn.Linear`` layers in
    ``Gemma4ClippableLinear`` (whose inner module is named ``.linear``);
    the language model uses raw ``nn.Linear`` directly named ``q_proj``
    / ``k_proj`` / ``v_proj`` / ``o_proj``.

    Earlier versions of this regex used ``\\.linear$``, which matched ONLY
    the vision/audio towers' inner linears — modules that are NEVER
    traversed during text-only forward passes — so every LoRA gradient was
    structurally zero (and ``clip_grad_norm_`` returned 0.0 every step,
    silently). The fixed regex below targets the language model's
    attention projections specifically.

    NOTE: When the base model is fully frozen AND gradient checkpointing is
    enabled (vocab cells in this codebase), the input embedding output has
    ``requires_grad=False``. Torch's gradient checkpointing then
    short-circuits the backward pass through every checkpointed
    transformer block — which silently zeros out the gradient to every
    LoRA adapter inside those blocks. The fix is to call
    ``enable_input_require_grads()`` on the PEFT model, which registers a
    forward hook on the input embeddings that forces
    ``output.requires_grad_(True)`` so backward can flow.
    See: https://github.com/huggingface/peft/issues/2826,
         https://github.com/huggingface/transformers/issues/42947,
         https://github.com/huggingface/transformers/issues/23170
    """
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Match only the LANGUAGE MODEL's attention projections (not the
        # vision_tower or audio_tower whose q_proj is a wrapper class).
        target_modules=r".*language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
    )
    peft_model = get_peft_model(model, cfg)
    if hasattr(peft_model, "enable_input_require_grads"):
        peft_model.enable_input_require_grads()
    elif hasattr(peft_model, "get_input_embeddings"):
        def _make_inputs_require_grad(_module, _input, output):
            output.requires_grad_(True)
        peft_model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)
    return peft_model


def extend_vocab(loaded: LoadedDecoder, n_new: int) -> int:
    """Add ``n_new`` rows to the embedding table for vocab-extension bridges.

    Returns the offset where the new rows start (== old vocab size).
    """
    old = loaded.vocab_size
    loaded.model.resize_token_embeddings(old + n_new)
    loaded.vocab_size = old + n_new
    return old
