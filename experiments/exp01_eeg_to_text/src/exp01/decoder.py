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
    """Wrap with PEFT LoRA on the standard attention projections.

    Gemma 4 wraps q/k/v/o ``nn.Linear`` layers in ``Gemma4ClippableLinear``,
    so PEFT's name match doesn't see a raw ``nn.Linear`` named ``q_proj``.
    The regex below targets the inner ``.linear`` module of those four
    projections specifically (not every nn.Linear in the model).
    """
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=r".*self_attn\.(q_proj|k_proj|v_proj|o_proj)(\.linear)?$",
    )
    return get_peft_model(model, cfg)


def extend_vocab(loaded: LoadedDecoder, n_new: int) -> int:
    """Add ``n_new`` rows to the embedding table for vocab-extension bridges.

    Returns the offset where the new rows start (== old vocab size).
    """
    old = loaded.vocab_size
    loaded.model.resize_token_embeddings(old + n_new)
    loaded.vocab_size = old + n_new
    return old
