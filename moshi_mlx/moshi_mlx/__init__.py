# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# flake8: noqa

"""
moshi_mlx - MLX Inference Codebase for Kyutai Audio Generation Models
=====================================================================

This package provides the MLX (Apple's Machine Learning framework) implementation
for running inference with Kyutai's audio generation models, including:

- Moshi: A real-time conversational AI model that can engage in spoken dialogue
- Mimi: A neural audio codec for high-quality audio compression and reconstruction
- Helium: A text-based language model for text generation tasks
- TTS (Text-to-Speech): Speech synthesis using Delayed Streams Modeling

The package is optimized for Apple Silicon (M1/M2/M3) hardware and leverages
MLX's efficient array operations and automatic differentiation capabilities.

Main Components:
----------------
- modules: Core neural network building blocks (transformers, convolutions, quantization)
- models: High-level model implementations (Lm, Mimi, LmGen, TTS)
- utils: Utility functions for sampling, loading, and other helper operations

Usage Example:
--------------
    import moshi_mlx
    from moshi_mlx import models, utils
    
    # Load and run a model
    model = models.Lm(models.config_v0_1())
    model.load_weights("model.safetensors")
"""

from . import modules, models, utils

# Package version following semantic versioning (MAJOR.MINOR.PATCH)
__version__ = "0.3.0"
