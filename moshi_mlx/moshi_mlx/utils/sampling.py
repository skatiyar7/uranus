# Taken from https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/sample_utils.py
# Copyright © 2023-2024 Apple Inc.

"""
Token Sampling Utilities
========================

This module provides various sampling strategies for selecting tokens from
probability distributions during autoregressive generation. These strategies
help control the diversity and quality of generated text and audio.

Sampling Strategies:
-------------------
- Temperature Sampling: Scales logits to control randomness
- Top-K Sampling: Samples only from the K most likely tokens
- Top-P (Nucleus) Sampling: Samples from tokens comprising top P probability mass
- Min-P Sampling: Samples from tokens above a minimum probability threshold

The Sampler class provides a unified interface for applying these strategies,
with sensible defaults for conversational AI applications.

Note: This code is adapted from Apple's MLX examples repository.
"""

from dataclasses import dataclass
from functools import partial

import mlx.core as mx


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def min_p_sampling(
    logits: mx.array,
    min_p: float,
    min_tokens_to_keep: int = 1,
    temperature=1.0,
) -> mx.array:
    """
    Apply min-p sampling to the logits.

    Min-p keeps all tokens that are above a minimum probability, scaled by the
    probability of the most likely token. As a result, the filter is more
    aggressive given a very high-probability token.
    
    This is a dynamic filtering method that adapts to the confidence of the
    model's predictions - when the model is very confident (high top probability),
    fewer alternatives are considered.

    Args:
        logits: The logits from the model's output [batch, vocab_size]
        min_p: Minimum token probability threshold (0.0-1.0). Typical values 
            are in the 0.01-0.2 range, comparably selective as setting 
            `top_p` in the 0.99-0.8 range.
        min_tokens_to_keep: Minimum number of tokens that cannot be filtered,
            ensuring at least this many options remain. Default: 1
        temperature: Temperature for softmax scaling. Default: 1.0

    Returns:
        Sampled token indices [batch]
    
    Raises:
        ValueError: If min_p is not in [0, 1] or min_tokens_to_keep < 1
    """
    if not (0 <= min_p <= 1.0):
        raise ValueError(
            f"`min_p` has to be a float in the [0, 1] interval, but is {min_p}"
        )
    if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
        raise ValueError(
            f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}"
        )
    # reference implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L531-L605  # noqa

    # Softmax probabilities
    probs = mx.softmax(logits * (1 / temperature), axis=-1)

    # Indices sorted in decreasing order
    sorted_indices = mx.argsort(-logits, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    # Top probability
    top_probs = mx.take_along_axis(probs, sorted_indices[..., :1], axis=-1)

    # Calculate the min_p threshold
    scaled_min_p = min_p * top_probs

    # Mask tokens that have a probability less than the scaled min_p
    tokens_to_remove = sorted_probs < scaled_min_p
    tokens_to_remove[..., :min_tokens_to_keep] = False

    # Create pool of tokens with probability less than scaled min_p
    selected_probs = mx.where(tokens_to_remove, 0, sorted_probs)

    # Return sampled token
    sorted_token = mx.random.categorical(mx.log(selected_probs), axis=-1)
    return mx.take_along_axis(sorted_indices, sorted_token[..., None], axis=-1).squeeze(
        -1
    )


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def top_k_sampling(
    logprobs: mx.array,
    top_k: int,
    temperature=1.0,
) -> mx.array:
    """
    Sample from only the top K tokens ranked by probability.
    
    This is a simple but effective filtering method that limits the sampling
    pool to the K most likely tokens, preventing the model from selecting
    very unlikely tokens.
    
    Args:
        logprobs: A tensor of log probabilities [batch, vocab_size]
        top_k: Number of top tokens to sample from. Must be > 0 and < vocab_size.
        temperature: Temperature for scaling logits. Default: 1.0
    
    Returns:
        Sampled token indices [batch]
    
    Raises:
        ValueError: If top_k is not in valid range
    """
    vocab_size = logprobs.shape[-1]
    if not isinstance(top_k, int) or not (0 < top_k < vocab_size):
        raise ValueError(
            f"`top_k` has to be an integer in the (0, {vocab_size}] interval,"
            f" but is {top_k}."
        )
    logprobs = logprobs * (1 / temperature)
    # Find indices of tokens NOT in top-k and mask them
    mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]
    masked_logprobs = mx.put_along_axis(
        logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1
    )
    return mx.random.categorical(masked_logprobs, axis=-1)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def top_p_sampling(logits: mx.array, top_p: float, temperature: float) -> mx.array:
    """
    Apply top-p (nucleus) sampling to logits.
    
    Top-p sampling selects from the smallest set of tokens whose cumulative
    probability exceeds the threshold p. This provides a dynamic vocabulary
    size that adapts to the model's confidence.
    
    When the model is confident (peaked distribution), fewer tokens are
    considered. When uncertain (flat distribution), more tokens are included.

    Args:
        logits: The logits from the model's output [batch, vocab_size]
        top_p: The cumulative probability threshold (0.0-1.0). Common values
            are 0.9-0.95 for diverse but coherent generation.
        temperature: Temperature parameter for softmax distribution reshaping.
    
    Returns:
        Sampled token indices [batch]
    """
    # referenced implementation from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L449-L460  # noqa
    probs = mx.softmax(logits * (1 / temperature), axis=-1)

    # sort probs in ascending order
    sorted_indices = mx.argsort(probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    # select tokens with cumulative probs below threshold
    top_probs = mx.where(
        cumulative_probs > 1 - top_p,
        sorted_probs,
        0,
    )

    sorted_token = mx.random.categorical(mx.log(top_probs), axis=-1)
    token = mx.take_along_axis(sorted_indices, sorted_token[..., None], axis=-1)

    return token.squeeze(-1)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def categorical_sampling(logits, temp):
    """
    Simple categorical sampling with temperature scaling.
    
    This is the most basic sampling method - it samples directly from the
    softmax distribution after applying temperature scaling.
    
    Args:
        logits: The logits from the model's output [batch, vocab_size]
        temp: Temperature for scaling. Higher = more random, lower = more deterministic.
    
    Returns:
        Sampled token indices [batch]
    """
    return mx.random.categorical(logits * (1 / temp))


@dataclass
class Sampler:
    """
    Unified token sampler supporting multiple sampling strategies.
    
    The Sampler class provides a convenient interface for sampling tokens
    from model output logits. It supports various sampling strategies that
    can be combined to control the diversity and quality of generation.
    
    Sampling Strategy Selection (in order of priority):
    1. If temp == 0: Greedy decoding (argmax)
    2. If top_k is set: Top-K sampling
    3. If top_p < 1.0: Top-P (nucleus) sampling
    4. If min_p > 0: Min-P sampling
    5. Otherwise: Standard categorical sampling
    
    Attributes:
        temp: Temperature for softmax scaling. 0 = greedy, higher = more random.
            Default: 0.8 (slightly random for natural variation)
        top_p: Cumulative probability threshold for nucleus sampling.
            Default: 0.95 (consider tokens comprising 95% of probability mass)
        top_k: Number of top tokens to consider. None = disabled.
            Default: None
        min_p: Minimum probability threshold (scaled by top probability).
            Default: 0.0 (disabled)
        min_tokens_to_keep: Minimum tokens to keep when using min_p.
            Default: 1
        logit_bias: Dictionary mapping token IDs to bias values to add to logits.
            Useful for encouraging or discouraging specific tokens.
            Default: None
    
    Example:
        >>> sampler = Sampler(temp=0.8, top_k=50)
        >>> token, logprobs = sampler(model_logits)
    """
    temp: float = 0.8
    top_p: float = 0.95
    top_k: int | None = None
    min_p: float = 0.0
    min_tokens_to_keep: int = 1
    logit_bias: dict[int, float] | None = None

    def __call__(self, logits: mx.array) -> tuple[mx.array, mx.array]:
        """
        Sample a token from the given logits.
        
        Args:
            logits: Model output logits [batch, vocab_size]
        
        Returns:
            Tuple of (sampled_tokens, log_probabilities):
            - sampled_tokens: Selected token indices [batch] as int32
            - log_probabilities: Log probabilities of all tokens [batch, vocab_size]
        """
        # Apply logit bias if specified
        if self.logit_bias:
            indices = mx.array(list(self.logit_bias.keys()))
            values = mx.array(list(self.logit_bias.values()))
            logits[:, indices] += values
        
        # Compute log probabilities for return value
        logprobs = logits - mx.logsumexp(logits)

        # Select sampling strategy
        if self.temp == 0:
            # Greedy decoding - select most likely token
            token = mx.argmax(logits, axis=-1)
        else:
            if self.top_k is not None and self.top_k > 0:
                token = top_k_sampling(logits, self.top_k, self.temp)
            elif self.top_p > 0 and self.top_p < 1.0:
                token = top_p_sampling(logits, self.top_p, self.temp)
            elif self.min_p != 0.0:
                token = min_p_sampling(
                    logits, self.min_p, self.min_tokens_to_keep, self.temp
                )
            else:
                token = categorical_sampling(logits, self.temp)

        return token.astype(mx.int32), logprobs
