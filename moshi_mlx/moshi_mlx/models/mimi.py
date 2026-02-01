# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Mimi Neural Audio Codec
=======================

This module implements the Mimi audio codec, a neural audio compression model
that converts audio waveforms to discrete tokens and back. Mimi is used by
Moshi for encoding user speech and decoding generated speech.

Architecture Overview:
---------------------
Mimi uses an encoder-decoder architecture with a transformer in the middle:

```
Audio (24kHz) --> SEANet Encoder --> Transformer --> Downsample --> RVQ --> Tokens
                                                                              |
Tokens --> RVQ Decode --> Upsample --> Transformer --> SEANet Decoder --> Audio
```

Key Components:
- SEANet Encoder/Decoder: Convolutional networks for audio compression
- Transformer: Adds temporal modeling between encoder and decoder
- Residual Vector Quantization (RVQ): Discretizes the latent space
- Downsampling/Upsampling: Adjusts frame rate between encoder and quantizer

The codec operates at:
- Sample rate: 24kHz
- Frame rate: 12.5 Hz (80ms frames)
- Codebook size: 2048 entries per level
- Configurable number of codebooks (typically 8-32)

Streaming Support:
-----------------
Mimi supports streaming operation through step-by-step encoding and decoding,
maintaining internal state for causal processing of audio chunks.
"""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from ..modules import (
    ConvDownsample1d,
    ConvTranspose1d,
    ConvTrUpsample1d,
    EuclideanCodebook,
    ProjectedTransformer,
    SeanetConfig,
    SeanetDecoder,
    SeanetEncoder,
    SplitResidualVectorQuantizer,
    TransformerConfig,
)


@dataclass
class MimiConfig:
    """
    Configuration for the Mimi audio codec.
    
    Attributes:
        channels: Number of audio channels (1 for mono)
        sample_rate: Audio sample rate in Hz (24000)
        frame_rate: Output frame rate in Hz (12.5)
        renormalize: Whether to renormalize audio
        seanet: Configuration for the SEANet encoder/decoder
        transformer: Configuration for the middle transformer
        quantizer_nq: Number of quantizer codebooks
        quantizer_bins: Number of entries per codebook (2048)
        quantizer_dim: Dimension of quantizer vectors (256)
    """
    channels: int
    sample_rate: float
    frame_rate: float
    renormalize: bool
    seanet: SeanetConfig
    transformer: TransformerConfig
    quantizer_nq: int
    quantizer_bins: int
    quantizer_dim: int


def mimi_202407(num_codebooks: int) -> MimiConfig:
    """
    Create the standard Mimi configuration from July 2024.
    
    This is the default configuration used with Moshi models.
    
    Args:
        num_codebooks: Number of RVQ codebooks to use (typically 8-32)
    
    Returns:
        MimiConfig with standard settings
    """
    seanet = SeanetConfig(
        dimension=512,
        channels=1,
        causal=True,
        nfilters=64,
        nresidual_layers=1,
        ratios=[8, 6, 5, 4],
        ksize=7,
        residual_ksize=3,
        last_ksize=3,
        dilation_base=2,
        pad_mode="constant",
        true_skip=True,
        compress=2,
    )
    transformer = TransformerConfig(
        d_model=seanet.dimension,
        num_heads=8,
        num_layers=8,
        causal=True,
        norm_first=True,
        bias_ff=False,
        bias_attn=False,
        layer_scale=0.01,
        positional_embedding="rope",
        use_conv_bias=True,
        gating=False,
        norm="layer_norm",
        context=250,
        max_period=10000,
        max_seq_len=8192,
        kv_repeat=1,
        dim_feedforward=2048,
        conv_layout=True,
        use_conv_block=False,
        cross_attention=False,
        conv_kernel_size=3,
    )
    return MimiConfig(
        channels=1,
        sample_rate=24000,
        frame_rate=12.5,
        renormalize=True,
        seanet=seanet,
        transformer=transformer,
        quantizer_nq=num_codebooks,
        quantizer_bins=2048,
        quantizer_dim=256,
    )


class Mimi(nn.Module):
    """
    Mimi Neural Audio Codec.
    
    A neural audio codec that compresses audio waveforms to discrete tokens
    using a combination of convolutional networks, transformers, and
    residual vector quantization.
    
    The codec supports both batch processing (encode/decode) and streaming
    processing (encode_step/decode_step) for real-time applications.
    
    Attributes:
        cfg: MimiConfig with codec parameters
        encoder: SEANet convolutional encoder
        decoder: SEANet convolutional decoder
        quantizer: Split residual vector quantizer
        encoder_transformer: Transformer after encoder
        decoder_transformer: Transformer before decoder
        downsample: Downsampling layer to match frame rate
        upsample: Upsampling layer from frame rate
    
    Example:
        >>> mimi = Mimi(mimi_202407(num_codebooks=8))
        >>> mimi.load_weights("mimi.safetensors")
        >>> codes = mimi.encode(audio)  # [B, num_codebooks, T]
        >>> reconstructed = mimi.decode(codes)  # [B, 1, samples]
    """

    def __init__(self, cfg: MimiConfig):
        super().__init__()
        dim = cfg.seanet.dimension
        self.cfg = cfg
        encoder_frame_rate = cfg.sample_rate / math.prod(cfg.seanet.ratios)
        downsample_stride = int(encoder_frame_rate / cfg.frame_rate)
        self.encoder = SeanetEncoder(cfg.seanet)
        self.decoder = SeanetDecoder(cfg.seanet)
        self.quantizer = SplitResidualVectorQuantizer(
            dim=cfg.quantizer_dim,
            input_dim=dim,
            output_dim=dim,
            nq=cfg.quantizer_nq,
            bins=cfg.quantizer_bins,
        )
        self.encoder_transformer = ProjectedTransformer(
            cfg.transformer,
            input_dim=dim,
            output_dims=[dim],
        )
        self.decoder_transformer = ProjectedTransformer(
            cfg.transformer,
            input_dim=dim,
            output_dims=[dim],
        )
        self.downsample = ConvDownsample1d(
            stride=downsample_stride,
            dim=dim,
            causal=True,
        )
        self.upsample = ConvTrUpsample1d(
            stride=downsample_stride,
            dim=dim,
            causal=True,
        )
        self.encoder_cache = self.encoder_transformer.make_cache()
        self.decoder_cache = self.decoder_transformer.make_cache()

    def reset_state(self):
        """Reset streaming state for encoder and decoder."""
        self.encoder.reset_state()
        self.decoder.reset_state()
        for c in self.decoder_cache:
            c.reset()
        for c in self.encoder_cache:
            c.reset()

    def reset_all(self):
        """Reset all streaming state including up/downsampling."""
        self.reset_state()
        self.upsample.reset_state()
        self.downsample.reset_state()

    def encode(self, xs: mx.array) -> mx.array:
        """
        Encode audio waveform to discrete tokens (batch mode).
        
        Resets internal state before encoding, suitable for processing
        complete audio files.
        
        Args:
            xs: Audio waveform [B, channels, samples]
        
        Returns:
            Audio tokens [B, num_codebooks, frames]
        """
        self.encoder.reset_state()
        for c in self.encoder_cache:
            c.reset()
        xs = self.encoder(xs)
        xs = self.encoder_transformer(xs, cache=self.encoder_cache)[0]
        xs = self.downsample(xs)
        return self.quantizer.encode(xs)

    def decode(self, xs: mx.array) -> mx.array:
        """
        Decode discrete tokens to audio waveform (batch mode).
        
        Resets internal state before decoding, suitable for processing
        complete token sequences.
        
        Args:
            xs: Audio tokens [B, num_codebooks, frames]
        
        Returns:
            Audio waveform [B, channels, samples]
        """
        self.decoder.reset_state()
        for c in self.decoder_cache:
            c.reset()
        xs = self.quantizer.decode(xs)
        xs = self.upsample(xs)
        xs = self.decoder_transformer(xs, cache=self.decoder_cache)[0]
        return self.decoder(xs)

    def encode_step(self, xs: mx.array) -> mx.array:
        """
        Encode a single audio chunk (streaming mode).
        
        Maintains internal state for causal streaming processing.
        Call reset_all() before starting a new stream.
        
        Args:
            xs: Audio chunk [B, channels, chunk_samples]
        
        Returns:
            Audio tokens for this chunk [B, num_codebooks, chunk_frames]
        """
        xs = self.encoder.step(xs)
        xs = self.encoder_transformer(xs, cache=self.encoder_cache)[0]
        xs = self.downsample.step(xs)
        xs = self.quantizer.encode(xs)
        return xs

    def decode_step(self, xs: mx.array) -> mx.array:
        """
        Decode a single token chunk (streaming mode).
        
        Maintains internal state for causal streaming processing.
        Call reset_all() before starting a new stream.
        
        Args:
            xs: Audio tokens [B, num_codebooks, chunk_frames]
        
        Returns:
            Audio chunk [B, channels, chunk_samples]
        """
        xs = self.quantizer.decode(xs)
        xs = self.upsample.step(xs)
        xs = self.decoder_transformer(xs, cache=self.decoder_cache)[0]
        xs = self.decoder.step(xs)
        return xs

    def warmup(self):
        """
        Warmup the codec by running a dummy encode/decode cycle.
        
        This compiles MLX kernels and allocates memory, reducing
        latency on the first real inference call.
        """
        pcm = mx.zeros((1, 1, 1920 * 4))
        codes = self.encode(pcm)
        pcm_out = self.decode(codes)
        mx.eval(pcm_out)

    @property
    def frame_rate(self) -> float:
        """Output frame rate in Hz (typically 12.5)."""
        return self.cfg.frame_rate

    @property
    def sample_rate(self) -> float:
        """Audio sample rate in Hz (typically 24000)."""
        return self.cfg.sample_rate

    def load_pytorch_weights(
        self,
        file: str,
        strict: bool = True,
    ) -> nn.Module:
        """
        Load weights from a PyTorch checkpoint file.
        
        Handles the conversion from PyTorch weight naming conventions
        to MLX naming conventions, including transposing convolution
        weights for the different layout expectations.
        
        Args:
            file: Path to the PyTorch checkpoint (.safetensors)
            strict: If True, raise error on missing/unexpected keys
        
        Returns:
            Self for method chaining
        """
        weights = []
        for k, v in mx.load(file).items():
            v: mx.array = v
            k: str = ".".join([s.removeprefix("_") for s in k.split(".")])
            if k.startswith("encoder.model."):
                k = k.replace("encoder.model.", "encoder.")
            if k.startswith("decoder.model."):
                k = k.replace("decoder.model.", "decoder.")
            if k.endswith(".in_proj_weight"):
                k = k.replace(".in_proj_weight", ".in_proj.weight")
            if k.endswith(".linear1.weight"):
                k = k.replace(".linear1.weight", ".gating.linear1.weight")
            if k.endswith(".linear2.weight"):
                k = k.replace(".linear2.weight", ".gating.linear2.weight")
            # Awfully hardcoded matching between the pytorch layers and their mlx equivalent :(
            for layerIdx, decoderIdx in enumerate([2, 5, 8, 11]):
                k = k.replace(
                    f"decoder.{decoderIdx}.", f"decoder.layers.{layerIdx}.upsample."
                )
                k = k.replace(
                    f"decoder.{decoderIdx + 1}.",
                    f"decoder.layers.{layerIdx}.residuals.0.",
                )
            for layerIdx, encoderIdx in enumerate([1, 4, 7, 10]):
                k = k.replace(
                    f"encoder.{encoderIdx}.", f"encoder.layers.{layerIdx}.residuals.0."
                )
                k = k.replace(
                    f"encoder.{encoderIdx + 2}.",
                    f"encoder.layers.{layerIdx}.downsample.",
                )

            k = k.replace("decoder.0.", "decoder.init_conv1d.")
            k = k.replace("decoder.14.", "decoder.final_conv1d.")
            k = k.replace("encoder.0.", "encoder.init_conv1d.")
            k = k.replace("encoder.14.", "encoder.final_conv1d.")
            k = k.replace(".block.1.", ".block.0.")
            k = k.replace(".block.3.", ".block.1.")

            # PyTorch layout for conv weights is outC, inC, kSize, for MLX it's outC, kSize, inC
            if (
                k.endswith(".conv.weight")
                or k.endswith(".output_proj.weight")
                or k.endswith(".input_proj.weight")
            ):
                v = v.swapaxes(-1, -2)
            # PyTorch layout for conv-transposed weights is inC, outC, kSize, for MLX it's outC, kSize, inC
            if k.endswith(".convtr.weight"):
                v = v.transpose(1, 2, 0)
            weights.append((k, v))
        m = self.load_weights(weights, strict=strict)

        def _filter_fn(module, name, _):
            if isinstance(module, EuclideanCodebook) and name == "initialized":
                module.update_in_place()
            if isinstance(module, ConvTranspose1d) and name == "weight":
                module.update_in_place()
            return True

        m.filter_and_map(_filter_fn)
        return m
