# Most of the code below comes from:
# https://github.com/ml-explore/mlx-examples/blob/6c2369e4b97f49fb5906ec46033497b39931b25d/llms/mlx_lm/models/base.py#L1
# Copyright © 2023-2024 Apple Inc.

"""
Key-Value Cache for Transformer Attention
=========================================

This module implements key-value caching mechanisms for efficient
autoregressive generation in transformer models. During generation,
the KV cache stores previously computed key and value tensors so they
don't need to be recomputed at each step.

Two cache implementations are provided:

1. KVCache: Standard growing cache that expands as needed. Suitable for
   sequences shorter than the maximum context length.

2. RotatingKVCache: Fixed-size cache that rotates old entries out when
   full. Suitable for very long sequences where memory is a concern.

The caches are designed to work with MLX's lazy evaluation model,
efficiently managing memory allocation and updates.

Note: This code is adapted from Apple's MLX examples repository.
"""

import inspect
from dataclasses import dataclass
from typing import Any

import mlx.core as mx


class KVCache:
    """
    Standard key-value cache for transformer attention.
    
    This cache grows dynamically as new key-value pairs are added,
    allocating memory in chunks (step size) for efficiency.
    
    The cache stores keys and values in the format [B, num_heads, seq_len, head_dim]
    and supports different dimensions for keys and values (useful for some
    attention variants).
    
    Attributes:
        n_kv_heads: Number of key-value heads
        k_head_dim: Dimension of key vectors
        v_head_dim: Dimension of value vectors
        keys: Cached key tensor
        values: Cached value tensor
        offset: Current position in the cache
        step: Allocation chunk size
    
    Example:
        >>> cache = KVCache(head_dim=64, n_kv_heads=8)
        >>> keys, values = cache.update_and_fetch(new_keys, new_values)
    """

    def __init__(self, head_dim, n_kv_heads):
        self.n_kv_heads = n_kv_heads
        if isinstance(head_dim, int):
            self.k_head_dim = self.v_head_dim = head_dim
        elif isinstance(head_dim, tuple) and len(head_dim) == 2:
            self.k_head_dim, self.v_head_dim = head_dim
        else:
            raise ValueError("head_dim must be an int or a tuple of two ints")
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = 256

    def update_and_fetch(self, keys, values) -> tuple[mx.array, mx.array]:
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B = keys.shape[0]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, self.n_kv_heads, n_steps * self.step, self.k_head_dim)
            v_shape = (B, self.n_kv_heads, n_steps * self.step, self.v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                assert self.values is not None
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        assert self.values is not None
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def reset(self):
        self.offset = 0
        self.keys = None
        self.values = None

    @property
    def state(self):
        return self.keys, self.values


class RotatingKVCache:
    """
    Fixed-size rotating key-value cache for long sequences.
    
    Unlike KVCache which grows indefinitely, RotatingKVCache maintains
    a fixed maximum size and rotates out old entries when full. This is
    useful for very long sequences where memory is a concern.
    
    The cache can optionally "keep" a certain number of initial tokens
    that are never rotated out (useful for system prompts or important
    context).
    
    Attributes:
        n_kv_heads: Number of key-value heads
        k_head_dim: Dimension of key vectors
        v_head_dim: Dimension of value vectors
        keep: Number of initial tokens to never rotate out
        max_size: Maximum cache size
        step: Allocation chunk size
    
    Example:
        >>> cache = RotatingKVCache(head_dim=64, n_kv_heads=8, max_size=2048)
        >>> keys, values = cache.update_and_fetch(new_keys, new_values)
    """

    def __init__(self, head_dim, n_kv_heads, max_size, keep=0, step=256):
        self.n_kv_heads = n_kv_heads
        if isinstance(head_dim, int):
            self.k_head_dim = self.v_head_dim = head_dim
        elif isinstance(head_dim, tuple) and len(head_dim) == 2:
            self.k_head_dim, self.v_head_dim = head_dim
        else:
            raise ValueError("head_dim must be an int or a tuple of two ints")
        self.keep = keep
        self.keys = None
        self.values = None
        self.offset = 0
        self.max_size = max_size
        self.step = step
        self._idx = 0

    def _trim(self, trim_size, v, append=None):
        to_cat = []
        if trim_size > 0:
            to_cat = [v[..., : self.keep, :], v[..., trim_size + self.keep :, :]]
        else:
            to_cat = [v]
        if append is not None:
            to_cat.append(append)
        return mx.concatenate(to_cat, axis=2)

    def update_and_fetch(self, keys, values) -> tuple[mx.array, mx.array]:
        prev = self.offset
        B, _, S = keys.shape[:3]

        # Prefill mode
        if S > 1:
            if self.keys is None:
                self.keys = keys
                self.values = values
            else:
                # The largest size is self.max_size + S - 1 to ensure
                # every token gets at least self.max_size context
                trim_size = self.keys.shape[2] - self.max_size + 1
                self.keys = self._trim(trim_size, self.keys, keys)
                self.values = self._trim(trim_size, self.values, values)
            self.offset += S
            self._idx = self.keys.shape[2]
            return self.keys, self.values

        # Generation mode
        # May not have hit the max size yet, so potentially
        # keep growing the cache
        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, self.n_kv_heads, new_size, self.k_head_dim)
            v_shape = (B, self.n_kv_heads, new_size, self.v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                assert self.values is not None
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self._idx = prev

        # Trim if needed
        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size

        # Rotate
        if self._idx == self.max_size:
            self._idx = self.keep

        # Assign
        self.keys[..., self._idx : self._idx + 1, :] = keys
        assert self.values is not None
        self.values[..., self._idx : self._idx + 1, :] = values
        self.offset += 1
        self._idx += 1

        # If the buffer is not full, slice off the end
        if self.offset < self.max_size:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        return self.keys, self.values

    def reset(self):
        self.offset = 0
        self._idx = 0
        self.keys = None
        self.values = None

    @property
    def state(self):
        return self.keys, self.values


@dataclass
class BaseModelArgs:
    """
    Base class for model argument dataclasses.
    
    Provides a from_dict class method that filters dictionary keys
    to only include those that match the dataclass fields.
    """

    @classmethod
    def from_dict(cls, params):
        """Create instance from dictionary, ignoring unknown keys."""
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def create_additive_causal_mask(N: int, offset: int = 0):
    """
    Create an additive causal attention mask.
    
    Creates a mask where position i can only attend to positions <= i.
    The mask uses large negative values (-1e9) for masked positions,
    which become ~0 after softmax.
    
    Args:
        N: Sequence length
        offset: Offset for the mask (for cached sequences)
    
    Returns:
        Causal mask of shape [N, offset + N]
    """
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    mask = linds[:, None] < rinds[None]
    return mask * -1e9


def create_attention_mask(h: mx.array, cache: Any | None = None):
    """
    Create an attention mask for the given input.
    
    For sequences longer than 1 token (prefill), creates a causal mask.
    For single tokens (generation), returns None (no mask needed with cache).
    
    Args:
        h: Input tensor [B, T, D]
        cache: Optional KV cache to determine offset
    
    Returns:
        Attention mask or None
    """
    T = h.shape[1]
    if T > 1:
        if cache is not None and cache[0] is not None:
            c = cache[0]
            if isinstance(c, RotatingKVCache):
                offset = min(c.max_size - 1, c.offset)
            else:
                offset = c.offset
        else:
            offset = 0
        mask = create_additive_causal_mask(T, offset)
        mask = mask.astype(h.dtype)
    else:
        mask = None
    return mask
