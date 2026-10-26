from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM


@dataclass
class EncoderOutput:
    states: torch.Tensor
    token_entropy: torch.Tensor


def cached_features(
    sequences: list[torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    if not sequences:
        raise ValueError("cannot pad an empty sequence list")
    values = [value.long() for value in sequences]
    input_ids = pad_sequence(values, batch_first=True, padding_value=0).to(device)
    lengths = torch.tensor([len(value) for value in values], device=device)
    positions = torch.arange(input_ids.shape[1], device=device)[None, :]
    return {
        "input_ids": input_ids,
        "attention_mask": (positions < lengths[:, None]).long(),
    }


def gather_last(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]


class Retriever(nn.Module):
    """The released method uses a single normalized last-token vector."""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    @classmethod
    def from_pretrained(cls, checkpoint, recipe=None, **kwargs):
        if recipe and (
            recipe.get("representation", "anchor") != "anchor"
            or recipe.get("latent_steps", 0)
            or recipe.get("slot_count", 0)
        ):
            raise ValueError("Only the single-vector main method is included.")
        lm = AutoModelForCausalLM.from_pretrained(
            checkpoint, trust_remote_code=False, dtype=torch.bfloat16, **kwargs
        )
        return cls(lm)

    @property
    def hidden_size(self):
        return int(self.lm.config.hidden_size)

    @property
    def backbone(self):
        return getattr(self.lm, getattr(self.lm, "base_model_prefix", "model"))

    def enable_gradient_checkpointing(self):
        self.lm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def encode_anchor_features(self, features):
        position_ids = features["attention_mask"].long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(features["attention_mask"] == 0, 0)
        hidden = self.backbone(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            position_ids=position_ids,
            return_dict=True,
            use_cache=False,
        ).last_hidden_state
        return F.normalize(
            gather_last(hidden, features["attention_mask"]).float(), dim=-1
        )

    def forward(self, features):
        anchor = self.encode_anchor_features(features)
        return EncoderOutput(anchor[:, None], anchor.new_empty(anchor.shape[0], 0))

    def adapt_documents(self, documents):
        return documents

    def save_inference(self, target, tokenizer):
        target = Path(target)
        target.mkdir(parents=True, exist_ok=False)
        self.lm.save_pretrained(target, safe_serialization=True)
        tokenizer.save_pretrained(target)


def primary_scores(states, documents, mode="anchor", alpha=0):
    if mode != "anchor":
        raise ValueError("The main method uses one query vector.")
    return states[:, 0] @ documents.T
