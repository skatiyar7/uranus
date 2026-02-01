# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# flake8: noqa
"""
Modules Subpackage
==================

This subpackage contains the core neural network building blocks used to
construct the Moshi and Mimi models. These modules are designed for
efficient inference on Apple Silicon using MLX.

Convolution Modules (conv.py):
-----------------------------
- Conv1d, ConvTranspose1d: Basic 1D convolution layers
- StreamableConv1d, StreamableConvTranspose1d: Streaming-capable convolutions
- NormConv1d, NormConvTranspose1d: Normalized convolution layers
- ConvDownsample1d, ConvTrUpsample1d: Strided convolutions for resampling

Quantization Modules (quantization.py):
--------------------------------------
- EuclideanCodebook: Vector quantization codebook with Euclidean distance
- VectorQuantization: Single-level vector quantizer
- ResidualVectorQuantization: Multi-level residual VQ (RVQ)
- SplitResidualVectorQuantizer: Split RVQ for efficient encoding

SEANet Modules (seanet.py):
--------------------------
- SeanetEncoder: Convolutional encoder for audio compression
- SeanetDecoder: Convolutional decoder for audio reconstruction
- SeanetResnetBlock: Residual blocks with dilated convolutions

Transformer Modules (transformer.py):
------------------------------------
- Transformer: Multi-layer transformer with attention and feedforward
- TransformerLayer: Single transformer layer with self/cross attention
- Attention: Multi-head self-attention with RoPE support
- CrossAttention: Cross-attention for conditioning
- ProjectedTransformer: Transformer with input/output projections

KV Cache Modules (kv_cache.py):
------------------------------
- KVCache: Standard key-value cache for autoregressive generation
- RotatingKVCache: Fixed-size rotating cache for long sequences

Conditioner Modules (conditioner.py):
------------------------------------
- ConditionProvider: Manages multiple conditioning sources
- LutConditioner: Lookup table based conditioning
- TensorConditioner: Tensor-based conditioning with cross-attention
"""

from .conv import (
    Conv1d,
    ConvTranspose1d,
    StreamableConv1d,
    StreamableConvTranspose1d,
    NormConv1d,
    NormConvTranspose1d,
    ConvDownsample1d,
    ConvTrUpsample1d,
)
from .quantization import SplitResidualVectorQuantizer, EuclideanCodebook
from .seanet import SeanetConfig, SeanetEncoder, SeanetDecoder
from .kv_cache import KVCache, RotatingKVCache
from .transformer import Transformer, TransformerConfig, ProjectedTransformer
