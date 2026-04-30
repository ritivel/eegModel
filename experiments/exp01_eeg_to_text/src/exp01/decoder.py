"""Gemma 4 decoder loader with frozen / LoRA modes and vocab extension."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from . import storage


@dataclass
class LoadedDecoder:
    model: nn.Module
    tokenizer: object
    embed_dim: int
    vocab_size: int


def load_decoder(model_id: str = "google/gemma-4-E2B-it", *, dtype: torch.dtype = torch.bfloat16) -> LoadedDecoder:
    """Load a Gemma 4 IT model + tokenizer from the hot HF cache."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=str(storage.HF_CACHE))
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=str(storage.HF_CACHE),
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    embed = model.get_input_embeddings()
    return LoadedDecoder(model=model, tokenizer=tok, embed_dim=embed.embedding_dim, vocab_size=embed.num_embeddings)


def freeze(model: nn.Module) -> nn.Module:
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


def attach_lora(model: nn.Module, *, r: int = 16, alpha: int = 32, dropout: float = 0.05) -> nn.Module:
    """Wrap with PEFT LoRA on the standard Gemma attention projections."""
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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
