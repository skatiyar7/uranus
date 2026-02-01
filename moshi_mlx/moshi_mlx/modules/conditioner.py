# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# flake8: noqa
"""
Conditioning Modules
====================

This module implements conditioning mechanisms for the Moshi language model,
allowing the model to be conditioned on various inputs like quality settings,
speaker embeddings, or other control signals.

Conditioning Types:
------------------
1. Text/Lookup Conditioning (LutConditioner): Uses a lookup table to map
   discrete values (like "very_good", "good", "bad") to embedding vectors.
   Useful for quality control or categorical conditioning.

2. Tensor Conditioning (TensorConditioner): Uses continuous tensor inputs
   (like speaker embeddings) with cross-attention. Useful for speaker
   conditioning or other continuous control signals.

Key Classes:
-----------
- ConditionAttributes: Container for all conditioning inputs
- ConditionTensor: Wrapper for computed condition embeddings
- ConditionProvider: Manages multiple conditioners and computes embeddings
- LutConditioner: Lookup table based conditioning
- TensorConditioner: Tensor-based conditioning with projection

The conditioning system is designed to be flexible, allowing multiple
conditioning sources to be combined and applied to the model.
"""

from dataclasses import dataclass, field
import typing as tp

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class TensorCondition:
    """
    Container for tensor-based conditioning input.
    
    Holds a tensor and its associated mask, where the mask indicates
    which positions contain valid data (useful for variable-length inputs).
    
    Attributes:
        tensor: Conditioning tensor [B, T, D]
        mask: Binary mask [B, T] indicating valid positions
    """

    tensor: mx.array
    mask: mx.array

    @staticmethod
    def from_tensor(tensor: mx.array):
        """Create a TensorCondition with all-ones mask."""
        B, T, _ = tensor.shape
        mask = mx.ones((B, T), dtype=mx.uint8)
        return TensorCondition(tensor, mask)

    @staticmethod
    def cat(conditions: tp.Sequence["TensorCondition"]) -> "TensorCondition":
        """Concatenate multiple TensorConditions along batch dimension."""
        assert conditions, "Cannot cat empty list."
        ref_tensor = conditions[0].tensor
        B, _, D = ref_tensor.shape
        assert B == 1
        B = len(conditions)
        T = max(condition.tensor.shape[1] for condition in conditions)
        mask = mx.zeros((B, T), dtype=mx.uint8)
        tensor = mx.zeros((B, T, D), dtype=ref_tensor.dtype)
        for b, condition in enumerate(conditions):
            tensor[b, : condition.tensor.shape[1], :] = condition.tensor[0]
            mask[b, : condition.mask.shape[1]] = condition.mask[0]
        return TensorCondition(tensor, mask)


@dataclass
class ConditionAttributes:
    """
    Container for all conditioning inputs to the model.
    
    Holds both text-based (lookup) and tensor-based conditioning inputs.
    This is the standard interface for passing conditioning information
    through the model.
    
    Attributes:
        text: Dictionary mapping conditioner names to text values
        tensor: Dictionary mapping conditioner names to TensorConditions
    """

    text: tp.Dict[str, tp.Optional[str]] = field(default_factory=dict)
    tensor: tp.Dict[str, TensorCondition] = field(default_factory=dict)

    @property
    def text_attributes(self) -> tp.Iterable[str]:
        """Names of text-based conditioning attributes."""
        return self.text.keys()

    @property
    def tensor_attributes(self) -> tp.Iterable[str]:
        """Names of tensor-based conditioning attributes."""
        return self.text.keys()

    @staticmethod
    def condition_types() -> tp.FrozenSet[str]:
        """Valid condition type names."""
        return frozenset(["text", "tensor"])

    def copy(self) -> "ConditionAttributes":
        """Create a shallow copy of the attributes."""
        return ConditionAttributes(dict(self.text), dict(self.tensor))


def create_sin_embedding(
    positions: mx.array,
    dim: int,
    max_period: float = 10000,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Create sinusoidal positional embedding, with shape `[B, T, C]`.

    Args:
        positions (torch.Tensor): LongTensor of positions.
        dim (int): Dimension of the embedding.
        max_period (float): Maximum period of the cosine/sine functions.
        dtype (torch.dtype or str): dtype to use to generate the embedding.
    Returns:
        torch.Tensor: Sinusoidal positional embedding.
    """
    # We aim for BTC format
    assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions.astype(dtype)
    adim = mx.arange(half_dim, dtype=dtype).reshape(1, 1, -1)
    phase = positions / (max_period ** (adim / (half_dim - 1)))
    return mx.concat([mx.cos(phase), mx.sin(phase)], axis=-1)


@dataclass
class TensorConditionerConfig:
    dim: int


class TensorConditioner(nn.Module):
    def __init__(self, output_dim: int, cfg: TensorConditionerConfig):
        super().__init__()

        self.output_proj = nn.Linear(cfg.dim, output_dim, bias=False)
        self.learnt_padding = mx.zeros((1, 1, output_dim))

    def condition(self, tc: TensorCondition) -> mx.array:
        cond, mask = tc.tensor, tc.mask
        cond = self.output_proj(cond)
        mask = mask.astype(cond.dtype)
        mask = mx.expand_dims(mask, axis=-1)
        cond = cond * mask + self.learnt_padding * (1 - mask)
        # sin embeddings
        pos = mx.arange(cond.shape[1])[None, :, None]
        emb = create_sin_embedding(pos, cond.shape[-1]).astype(cond.dtype)
        return cond + emb


@dataclass
class LutConditionerConfig:
    n_bins: int
    dim: int
    tokenizer: str
    possible_values: dict[str, int]


class LutConditioner(nn.Module):
    def __init__(self, output_dim: int, cfg: LutConditionerConfig):
        super().__init__()

        if cfg.tokenizer != "noop":
            raise ValueError(f"unsupported tokenizer {cfg.tokenizer}")

        self.embed = nn.Embedding(cfg.n_bins + 1, cfg.dim)
        self.output_proj = nn.Linear(cfg.dim, output_dim, bias=False)
        self.learnt_padding = mx.zeros((1, 1, output_dim))
        self.possible_values = {v: i for i, v in enumerate(cfg.possible_values)}

    def condition(self, value: str) -> mx.array:
        idx = self.possible_values.get(value, None)
        if idx is None:
            raise ValueError(
                f"unknown value {value}, possible-values: {self.possible_values}"
            )
        idx = mx.array([idx])
        return self.output_proj(self.embed(idx))


@dataclass
class ConditionTensor:
    tensor: mx.array


class ConditionProvider(nn.Module):
    def __init__(self, output_dim: int, cfg: dict):
        self.conditioners = {}
        for name, c in cfg.items():
            if isinstance(c, LutConditionerConfig):
                cond = LutConditioner(output_dim, c)
            elif isinstance(c, TensorConditionerConfig):
                cond = TensorConditioner(output_dim, c)
            else:
                raise ValueError(f"unsupported config type {type(c)}")
            self.conditioners[name] = cond

    def condition_tensor(self, name: str, value: str) -> ConditionTensor:
        if name not in self.conditioners:
            raise ValueError(f"unsupported conditioner {name}")
        tensor = self.conditioners[name].condition(value)
        return ConditionTensor(tensor)


def dropout_tensor(condition: TensorCondition) -> TensorCondition:
    """Utility function for nullifying a WavCondition object."""
    return TensorCondition(
        tensor=mx.zeros_like(condition.tensor), mask=mx.zeros_like(condition.mask)
    )


def dropout_condition_(
    sample: ConditionAttributes, condition_type: str, condition: str
) -> None:
    """Utility function for nullifying an attribute inside a ConditionAttributes object.
    Works in-place.
    """
    valid_conditions = ConditionAttributes.condition_types()
    if condition_type not in valid_conditions:
        raise ValueError(
            "dropout_condition got an unexpected condition type!"
            f" expected one of {valid_conditions} but got '{condition_type}'"
        )

    if condition not in getattr(sample, condition_type):
        raise ValueError(
            "dropout_condition received an unexpected condition!"
            f" expected tensor={sample.tensor.keys()} and text={sample.text.keys()}"
            f" but got '{condition}' of type '{condition_type}'!"
        )

    if condition_type == "tensor":
        tensor_condition = sample.tensor[condition]
        sample.tensor[condition] = dropout_tensor(tensor_condition)
    elif condition_type == "text":
        sample.text[condition] = None
    else:
        assert False


def dropout_all_conditions(
    attributes: tp.Sequence[ConditionAttributes],
) -> list[ConditionAttributes]:
    """
    Args:
        attributes (list[ConditionAttributes]): All conditions attributes.
    Returns:
        list[ConditionAttributes]: Same with all conditions dropped.
    """
    attributes = [attribute.copy() for attribute in attributes]
    for condition_type in ConditionAttributes.condition_types():
        for attribute in attributes:
            for condition in getattr(attribute, condition_type):
                dropout_condition_(attribute, condition_type, condition)
    return attributes
