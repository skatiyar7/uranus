# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# flake8: noqa
"""
Models Subpackage
=================

This subpackage contains the high-level model implementations for the
moshi_mlx inference codebase. It provides the main model classes and
configuration functions for:

- Lm: The main language model class supporting text and audio generation
- LmGen: Generation wrapper that handles delayed streams and sampling
- Mimi: Neural audio codec for encoding/decoding audio to/from tokens
- TTS: Text-to-speech model using Delayed Streams Modeling

Model Configurations:
--------------------
- config_v0_1(): Original Moshi v0.1 configuration (7B parameters)
- config1b_202412(): 1B parameter model from December 2024
- config1b_202412_16rvq(): 1B model with 16 RVQ codebooks
- config_helium_1_preview_2b(): Helium text-only model (2B parameters)

The models are designed to work with MLX for efficient inference on
Apple Silicon hardware.
"""

from .lm import (
    Lm,
    LmConfig,
    config_v0_1,
    config1b_202412,
    config1b_202412_16rvq,
    config_helium_1_preview_2b,
)
from .generate import LmGen
from .mimi import mimi_202407, MimiConfig
