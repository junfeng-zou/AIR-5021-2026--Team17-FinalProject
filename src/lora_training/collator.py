"""Vendored from prismatic (openvla): pad batches for action prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

IGNORE_INDEX = -100


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        pixel_values = [instance["pixel_values"] for instance in instances]

        assert self.padding_side == "right", f"Invalid padding_side={self.padding_side}"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        input_ids = input_ids[:, : self.model_max_length]
        labels = labels[:, : self.model_max_length]

        attention_mask = input_ids.ne(self.pad_token_id)

        assert all(pv is not None for pv in pixel_values), "VLA examples require pixel_values"

        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values_out = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values_out = {
                k: torch.stack([pv[k] for pv in pixel_values]) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported pixel_values type: {type(pixel_values[0])}")

        return {
            "pixel_values": pixel_values_out,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
