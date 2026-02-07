# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Mimi Neural Audio Codec
=======================

This module implements the Mimi audio codec, a neural audio compression model
that converts audio waveforms to discrete tokens and back. Mimi is used by
Moshi for encoding user speech and decoding generated speech.

=============================================================================
ROLE IN THE MOSHI SYSTEM
=============================================================================

Mimi is the AUDIO TOKENIZER for Moshi. It serves two critical functions:

1. ENCODING (User Speech → Tokens)
   User audio → Mimi.encode_step() → audio tokens → Moshi LM input
   
2. DECODING (Tokens → Moshi Speech)
   Moshi LM output → audio tokens → Mimi.decode_step() → Moshi audio

This enables Moshi to work with discrete tokens instead of raw audio,
making it compatible with transformer-based language modeling.

=============================================================================
ARCHITECTURE OVERVIEW
=============================================================================

Mimi uses an encoder-decoder architecture with a transformer in the middle:

ENCODING PATH:
```
Audio (24kHz, mono)
    ↓
SEANet Encoder (convolutional, causal)
    ↓ [B, 512, T_enc]
Encoder Transformer (8 layers, causal)
    ↓ [B, 512, T_enc]
Convolutional Downsampler (stride=8)
    ↓ [B, 512, T_frame]
Split RVQ Encoder
    ↓
Tokens [B, num_codebooks, T_frame]
```

DECODING PATH:
```
Tokens [B, num_codebooks, T_frame]
    ↓
Split RVQ Decoder
    ↓ [B, 512, T_frame]
Convolutional Upsampler (stride=8)
    ↓ [B, 512, T_enc]
Decoder Transformer (8 layers, causal)
    ↓ [B, 512, T_enc]
SEANet Decoder (convolutional, causal)
    ↓
Audio (24kHz, mono)
```

=============================================================================
KEY COMPONENTS
=============================================================================

1. SEANet ENCODER/DECODER
   - Convolutional networks for audio compression/decompression
   - Ratios [8, 6, 5, 4] give total downsampling of 960x
   - 24000 Hz / 960 = 25 Hz intermediate frame rate
   - Causal for streaming operation

2. TRANSFORMER
   - 8 layers, 8 heads, d_model=512
   - Adds temporal modeling between encoder and decoder
   - Uses RoPE positional embeddings
   - Causal for streaming

3. RESIDUAL VECTOR QUANTIZATION (RVQ)
   - Discretizes the continuous latent space
   - Multiple codebooks (typically 8-32) for hierarchical representation
   - First codebook captures semantic content
   - Later codebooks capture acoustic details
   - 2048 entries per codebook

4. DOWNSAMPLING/UPSAMPLING
   - Additional 8x down/up between transformer and RVQ
   - Final frame rate: 25 Hz / 8 = 12.5 Hz
   - Matches Moshi's 12.5 Hz generation rate

=============================================================================
OPERATING PARAMETERS
=============================================================================

- Sample rate: 24,000 Hz
- Frame rate: 12.5 Hz (80ms per frame)
- Samples per frame: 24000 / 12.5 = 1920 samples
- Codebook size: 2048 entries per level
- Typical codebooks: 8 or 16 (for Moshi) or 32 (for high-quality TTS)
- Latent dimension: 512

=============================================================================
STREAMING SUPPORT
=============================================================================

Mimi supports streaming operation through step-by-step encoding and decoding:

    mimi.reset_all()  # Reset all internal state
    
    for audio_chunk in audio_stream:
        # audio_chunk: [1, 1, 1920] (80ms at 24kHz)
        tokens = mimi.encode_step(audio_chunk)  # [1, 8, 1]
        
        # ... process tokens through Moshi ...
        
        output_audio = mimi.decode_step(output_tokens)  # [1, 1, 1920]

The causal architecture ensures that:
- Each output frame depends only on current and past input
- No future information is used (essential for real-time)
- Internal state is maintained between calls

=============================================================================
RELATIONSHIP TO MOSHI'S TOKEN STREAMS
=============================================================================

In full-duplex Moshi:
- User audio → Mimi.encode_step() → 8 codebooks (user's speech tokens)
- These 8 codebooks are fed as INPUT to the Moshi LM
- Moshi LM generates 8 codebooks (Moshi's speech tokens)
- Moshi's tokens → Mimi.decode_step() → Moshi audio

Total: 16 or 32 audio codebooks in the LM (8 or 16 user + 8 or 16 Moshi)
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
    
    ==========================================================================
    CONFIGURATION PARAMETERS
    ==========================================================================
    
    This dataclass holds all parameters needed to construct a Mimi codec.
    The default configuration (mimi_202407) provides settings optimized
    for use with Moshi.
    
    Attributes:
        channels: Number of audio channels (1 for mono, 2 for stereo).
                 Moshi uses mono audio (channels=1).
        
        sample_rate: Audio sample rate in Hz.
                    Standard is 24000 Hz for Mimi/Moshi.
        
        frame_rate: Output frame rate in Hz after all downsampling.
                   12.5 Hz means one token frame per 80ms of audio.
                   This matches Moshi's generation rate.
        
        renormalize: Whether to renormalize audio during processing.
                    Helps with varying input levels.
        
        seanet: Configuration for the SEANet encoder/decoder.
               Controls the convolutional architecture, downsampling ratios,
               kernel sizes, and other conv-specific parameters.
        
        transformer: Configuration for the middle transformer.
                    Controls layers, heads, dimension, attention settings.
        
        quantizer_nq: Number of RVQ codebooks (quantizer levels).
                     More codebooks = higher quality but more tokens.
                     Typical: 8 for Moshi, 32 for high-quality TTS.
        
        quantizer_bins: Number of entries per codebook.
                       2048 is standard, giving 11 bits per codebook.
        
        quantizer_dim: Dimension of quantizer vectors.
                      256 is typical, projects from 512-dim latent.
    
    ==========================================================================
    FRAME RATE CALCULATION
    ==========================================================================
    
    The frame rate is determined by:
    1. SEANet downsampling: prod([8, 6, 5, 4]) = 960x
    2. Additional downsampling: 2x (stride=2 in standard config)
    3. Total: 960 * 2 = 1920x
    4. Frame rate: 24000 / 1920 = 12.5 Hz
    
    Breakdown:
    1. SEANet: 24000 / 960 = 25 Hz (intermediate frame rate)
    2. Downsample: 25 / 2 = 12.5 Hz (final frame rate)
    
    The downsample_stride is computed as:
        encoder_frame_rate / target_frame_rate = 25 / 12.5 = 2
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
    
    This is the default configuration used with Moshi models. It provides
    a good balance between audio quality and computational efficiency.
    
    ==========================================================================
    ARCHITECTURE DETAILS
    ==========================================================================
    
    SEANet Encoder/Decoder:
    - Dimension: 512
    - Downsampling ratios: [8, 6, 5, 4] → 960x total
    - Kernel size: 7 (main), 3 (residual)
    - Dilation base: 2 (for increasing receptive field)
    - Causal: True (for streaming)
    
    Transformer:
    - Layers: 8
    - Heads: 8
    - Dimension: 512
    - FFN dimension: 2048
    - Positional embedding: RoPE
    - Causal: True (for streaming)
    
    Quantizer:
    - Codebook size: 2048 entries
    - Vector dimension: 256
    - Number of codebooks: configurable (typically 8 or 32)
    
    ==========================================================================
    BITRATE CALCULATION
    ==========================================================================
    
    With 8 codebooks at 12.5 Hz:
    - Bits per frame: 8 codebooks × 11 bits = 88 bits
    - Bitrate: 88 × 12.5 = 1100 bps = 1.1 kbps
    
    With 32 codebooks:
    - Bitrate: 32 × 11 × 12.5 = 4400 bps = 4.4 kbps
    
    ==========================================================================
    
    Args:
        num_codebooks: Number of RVQ codebooks to use.
                      8 for Moshi (lower latency)
                      32 for high-quality TTS
    
    Returns:
        MimiConfig with standard settings for the specified codebook count.
    """
    # =========================================================================
    # SEANET CONFIGURATION
    # =========================================================================
    # SEANet is the convolutional encoder/decoder backbone
    seanet = SeanetConfig(
        dimension=512,              # Latent dimension
        channels=1,                 # Mono audio
        causal=True,                # Essential for streaming
        nfilters=64,                # Base number of filters
        nresidual_layers=1,         # Residual blocks per layer
        ratios=[8, 6, 5, 4],        # Downsampling ratios (total 960x)
        ksize=7,                    # Main kernel size
        residual_ksize=3,           # Residual block kernel size
        last_ksize=3,               # Final layer kernel size
        dilation_base=2,            # Dilation increases receptive field
        pad_mode="constant",        # Padding mode for causal conv
        true_skip=True,             # Use true skip connections
        compress=2,                 # Compression factor for filters
    )
    
    # =========================================================================
    # TRANSFORMER CONFIGURATION
    # =========================================================================
    # Transformer adds temporal modeling between encoder and decoder
    transformer = TransformerConfig(
        d_model=seanet.dimension,   # Match SEANet dimension (512)
        num_heads=8,                # Attention heads
        num_layers=8,               # Transformer layers
        causal=True,                # Essential for streaming
        norm_first=True,            # Pre-norm (more stable)
        bias_ff=False,              # No bias in FFN
        bias_attn=False,            # No bias in attention
        layer_scale=0.01,           # Small layer scale for stability
        positional_embedding="rope", # Rotary position embeddings
        use_conv_bias=True,         # Bias in conv layers
        gating=False,               # No GLU gating
        norm="layer_norm",          # LayerNorm (not RMSNorm)
        context=250,                # Context window (250 frames = 20s)
        max_period=10000,           # RoPE max period
        max_seq_len=8192,           # Maximum sequence length
        kv_repeat=1,                # No GQA
        dim_feedforward=2048,       # FFN dimension (4x d_model)
        conv_layout=True,           # Use conv layout for efficiency
        use_conv_block=False,       # No conv blocks in transformer
        cross_attention=False,      # No cross-attention
        conv_kernel_size=3,         # Conv kernel size
    )
    
    return MimiConfig(
        channels=1,                 # Mono audio
        sample_rate=24000,          # 24kHz sample rate
        frame_rate=12.5,            # 12.5 Hz frame rate (80ms frames)
        renormalize=True,           # Normalize audio levels
        seanet=seanet,
        transformer=transformer,
        quantizer_nq=num_codebooks, # Number of RVQ levels
        quantizer_bins=2048,        # Codebook size (11 bits)
        quantizer_dim=256,          # Quantizer vector dimension
    )


class Mimi(nn.Module):
    """
    Mimi Neural Audio Codec.
    
    A neural audio codec that compresses audio waveforms to discrete tokens
    using a combination of convolutional networks, transformers, and
    residual vector quantization.
    
    ==========================================================================
    ROLE IN MOSHI
    ==========================================================================
    
    Mimi is the BRIDGE between continuous audio and discrete tokens:
    
    1. ENCODING: Converts user's speech to tokens for the LM
       user_audio → encode_step() → tokens → Moshi LM input
    
    2. DECODING: Converts LM output tokens back to audio
       Moshi LM output → tokens → decode_step() → moshi_audio
    
    ==========================================================================
    ARCHITECTURE
    ==========================================================================
    
    ENCODING PATH:
        Audio [B, 1, samples]
            ↓ SEANet Encoder (conv, 960x downsample)
        Latent [B, 512, T_enc]
            ↓ Encoder Transformer (8 layers)
        Latent [B, 512, T_enc]
            ↓ Downsample (2x)
        Latent [B, 512, T_frame]
            ↓ RVQ Encode
        Tokens [B, num_codebooks, T_frame]
    
    DECODING PATH:
        Tokens [B, num_codebooks, T_frame]
            ↓ RVQ Decode
        Latent [B, 512, T_frame]
            ↓ Upsample (2x)
        Latent [B, 512, T_enc]
            ↓ Decoder Transformer (8 layers)
        Latent [B, 512, T_enc]
            ↓ SEANet Decoder (conv, 960x upsample)
        Audio [B, 1, samples]
    
    ==========================================================================
    STREAMING VS BATCH MODES
    ==========================================================================
    
    BATCH MODE (encode/decode):
    - Processes complete audio files
    - Resets internal state before processing
    - Use for offline processing
    
    STREAMING MODE (encode_step/decode_step):
    - Processes audio chunk by chunk
    - Maintains internal state between calls
    - Use for real-time applications
    - Call reset_all() before starting a new stream
    
    ==========================================================================
    ATTRIBUTES
    ==========================================================================
    
    Configuration:
        cfg: MimiConfig with codec parameters
    
    Encoder Path:
        encoder: SEANet convolutional encoder
        encoder_transformer: Transformer after encoder
        downsample: Convolutional downsampler
        encoder_cache: KV cache for encoder transformer
    
    Decoder Path:
        decoder: SEANet convolutional decoder
        decoder_transformer: Transformer before decoder
        upsample: Convolutional upsampler
        decoder_cache: KV cache for decoder transformer
    
    Quantization:
        quantizer: Split residual vector quantizer
    
    ==========================================================================
    EXAMPLE USAGE
    ==========================================================================
    
    Batch mode:
        >>> mimi = Mimi(mimi_202407(num_codebooks=8))
        >>> mimi.load_pytorch_weights("mimi.safetensors")
        >>> 
        >>> # Encode audio to tokens
        >>> audio = load_audio("speech.wav")  # [1, 1, samples]
        >>> codes = mimi.encode(audio)  # [1, 8, frames]
        >>> 
        >>> # Decode tokens back to audio
        >>> reconstructed = mimi.decode(codes)  # [1, 1, samples]
    
    Streaming mode:
        >>> mimi.reset_all()
        >>> for chunk in audio_stream:
        ...     # chunk: [1, 1, 1920] (80ms at 24kHz)
        ...     tokens = mimi.encode_step(chunk)  # [1, 8, 1]
        ...     output = mimi.decode_step(tokens)  # [1, 1, 1920]
    """

    def __init__(self, cfg: MimiConfig):
        """
        Initialize the Mimi codec.
        
        Constructs all components of the encoder-decoder architecture:
        - SEANet encoder and decoder
        - Encoder and decoder transformers
        - Downsampler and upsampler
        - Split residual vector quantizer
        
        Args:
            cfg: MimiConfig with all codec parameters
        """
        super().__init__()
        dim = cfg.seanet.dimension  # 512
        self.cfg = cfg
        
        # =====================================================================
        # COMPUTE DOWNSAMPLE STRIDE
        # =====================================================================
        # SEANet downsamples by prod(ratios) = 8*6*5*4 = 960
        # This gives encoder_frame_rate = 24000/960 = 25 Hz
        # We need additional downsampling to reach target frame_rate (12.5 Hz)
        encoder_frame_rate = cfg.sample_rate / math.prod(cfg.seanet.ratios)
        downsample_stride = int(encoder_frame_rate / cfg.frame_rate)  # 25/12.5 = 2
        
        # =====================================================================
        # ENCODER PATH
        # =====================================================================
        # SEANet encoder: audio → latent (960x downsample)
        self.encoder = SeanetEncoder(cfg.seanet)
        
        # Encoder transformer: adds temporal modeling
        self.encoder_transformer = ProjectedTransformer(
            cfg.transformer,
            input_dim=dim,
            output_dims=[dim],
        )
        
        # Additional downsampling to reach target frame rate
        self.downsample = ConvDownsample1d(
            stride=downsample_stride,
            dim=dim,
            causal=True,
        )
        
        # =====================================================================
        # DECODER PATH
        # =====================================================================
        # SEANet decoder: latent → audio (960x upsample)
        self.decoder = SeanetDecoder(cfg.seanet)
        
        # Decoder transformer: adds temporal modeling
        self.decoder_transformer = ProjectedTransformer(
            cfg.transformer,
            input_dim=dim,
            output_dims=[dim],
        )
        
        # Upsampling from frame rate to encoder rate
        self.upsample = ConvTrUpsample1d(
            stride=downsample_stride,
            dim=dim,
            causal=True,
        )
        
        # =====================================================================
        # QUANTIZER
        # =====================================================================
        # Split RVQ: discretizes the latent space
        # "Split" means first codebook is separate (semantic) from rest (acoustic)
        self.quantizer = SplitResidualVectorQuantizer(
            dim=cfg.quantizer_dim,      # 256
            input_dim=dim,              # 512
            output_dim=dim,             # 512
            nq=cfg.quantizer_nq,        # Number of codebooks
            bins=cfg.quantizer_bins,    # 2048 entries per codebook
        )
        
        # =====================================================================
        # KV CACHES FOR STREAMING
        # =====================================================================
        # These caches store key-value pairs for efficient streaming inference
        self.encoder_cache = self.encoder_transformer.make_cache()
        self.decoder_cache = self.decoder_transformer.make_cache()

    def reset_state(self):
        """
        Reset streaming state for encoder and decoder transformers.
        
        Clears the KV caches and resets the SEANet encoder/decoder state.
        Call this when starting a new audio stream but keeping the
        up/downsampler state (rare use case).
        
        For most cases, use reset_all() instead.
        """
        self.encoder.reset_state()
        self.decoder.reset_state()
        for c in self.decoder_cache:
            c.reset()
        for c in self.encoder_cache:
            c.reset()

    def reset_all(self):
        """
        Reset ALL streaming state including up/downsampling.
        
        This is the PRIMARY reset method for streaming operation.
        Call this before starting a new audio stream to ensure
        clean state.
        
        Resets:
        - SEANet encoder/decoder internal buffers
        - Transformer KV caches
        - Upsampler/downsampler internal state
        
        Example:
            >>> mimi.reset_all()
            >>> for chunk in new_audio_stream:
            ...     tokens = mimi.encode_step(chunk)
        """
        self.reset_state()
        self.upsample.reset_state()
        self.downsample.reset_state()

    def encode(self, xs: mx.array) -> mx.array:
        """
        Encode audio waveform to discrete tokens (batch mode).
        
        Processes a complete audio file, resetting internal state first.
        Use this for offline processing of complete audio files.
        
        =======================================================================
        ENCODING PIPELINE
        =======================================================================
        
        1. Reset encoder state (for clean processing)
        2. SEANet encoder: audio → latent (960x downsample)
        3. Encoder transformer: temporal modeling
        4. Downsample: latent → frame rate (2x)
        5. RVQ encode: continuous → discrete tokens
        
        =======================================================================
        
        Args:
            xs: Audio waveform [B, channels, samples]
                - B: batch size
                - channels: 1 for mono
                - samples: number of audio samples
        
        Returns:
            Audio tokens [B, num_codebooks, frames]
            - num_codebooks: typically 8 or 32
            - frames: samples / (960 * 2) = samples / 1920
        
        Example:
            >>> audio = mx.zeros((1, 1, 48000))  # 2 seconds at 24kHz
            >>> tokens = mimi.encode(audio)  # [1, 8, 25] (25 frames)
        """
        # Reset state for clean batch processing
        self.encoder.reset_state()
        for c in self.encoder_cache:
            c.reset()
        
        # Encoding pipeline
        xs = self.encoder(xs)                                    # [B, 512, T_enc]
        xs = self.encoder_transformer(xs, cache=self.encoder_cache)[0]  # [B, 512, T_enc]
        xs = self.downsample(xs)                                 # [B, 512, T_frame]
        return self.quantizer.encode(xs)                         # [B, nq, T_frame]

    def decode(self, xs: mx.array) -> mx.array:
        """
        Decode discrete tokens to audio waveform (batch mode).
        
        Processes a complete token sequence, resetting internal state first.
        Use this for offline processing of complete token sequences.
        
        =======================================================================
        DECODING PIPELINE
        =======================================================================
        
        1. Reset decoder state (for clean processing)
        2. RVQ decode: discrete tokens → continuous latent
        3. Upsample: frame rate → encoder rate (2x)
        4. Decoder transformer: temporal modeling
        5. SEANet decoder: latent → audio (960x upsample)
        
        =======================================================================
        
        Args:
            xs: Audio tokens [B, num_codebooks, frames]
        
        Returns:
            Audio waveform [B, channels, samples]
            - samples: frames * 1920
        
        Example:
            >>> tokens = mx.zeros((1, 8, 25), dtype=mx.int32)  # 25 frames
            >>> audio = mimi.decode(tokens)  # [1, 1, 48000] (2 seconds)
        """
        # Reset state for clean batch processing
        self.decoder.reset_state()
        for c in self.decoder_cache:
            c.reset()
        
        # Decoding pipeline
        xs = self.quantizer.decode(xs)                           # [B, 512, T_frame]
        xs = self.upsample(xs)                                   # [B, 512, T_enc]
        xs = self.decoder_transformer(xs, cache=self.decoder_cache)[0]  # [B, 512, T_enc]
        return self.decoder(xs)                                  # [B, 1, samples]

    def encode_step(self, xs: mx.array) -> mx.array:
        """
        Encode a single audio chunk (streaming mode).
        
        Processes one chunk of audio, maintaining internal state for
        causal streaming. Call reset_all() before starting a new stream.
        
        =======================================================================
        STREAMING CONSIDERATIONS
        =======================================================================
        
        - Each call processes exactly one chunk
        - Internal state is maintained between calls
        - Output depends only on current and past input (causal)
        - Chunk size should be 1920 samples (80ms at 24kHz) for one frame
        
        =======================================================================
        
        Args:
            xs: Audio chunk [B, channels, chunk_samples]
                Typically [1, 1, 1920] for one frame
        
        Returns:
            Audio tokens for this chunk [B, num_codebooks, chunk_frames]
            Typically [1, 8, 1] for one frame
        
        Example:
            >>> mimi.reset_all()
            >>> for i in range(100):
            ...     chunk = get_audio_chunk()  # [1, 1, 1920]
            ...     tokens = mimi.encode_step(chunk)  # [1, 8, 1]
        """
        xs = self.encoder.step(xs)                               # [B, 512, T_enc_chunk]
        xs = self.encoder_transformer(xs, cache=self.encoder_cache)[0]
        xs = self.downsample.step(xs)                            # [B, 512, T_frame_chunk]
        xs = self.quantizer.encode(xs)                           # [B, nq, T_frame_chunk]
        return xs

    def decode_step(self, xs: mx.array) -> mx.array:
        """
        Decode a single token chunk (streaming mode).
        
        Processes one chunk of tokens, maintaining internal state for
        causal streaming. Call reset_all() before starting a new stream.
        
        =======================================================================
        STREAMING CONSIDERATIONS
        =======================================================================
        
        - Each call processes exactly one chunk
        - Internal state is maintained between calls
        - Output depends only on current and past input (causal)
        - Chunk size should be 1 frame for real-time operation
        
        =======================================================================
        
        Args:
            xs: Audio tokens [B, num_codebooks, chunk_frames]
                Typically [1, 8, 1] for one frame
        
        Returns:
            Audio chunk [B, channels, chunk_samples]
            Typically [1, 1, 1920] for one frame (80ms at 24kHz)
        
        Example:
            >>> mimi.reset_all()
            >>> for tokens in token_stream:
            ...     # tokens: [1, 8, 1]
            ...     audio = mimi.decode_step(tokens)  # [1, 1, 1920]
            ...     play(audio)
        """
        xs = self.quantizer.decode(xs)                           # [B, 512, T_frame_chunk]
        xs = self.upsample.step(xs)                              # [B, 512, T_enc_chunk]
        xs = self.decoder_transformer(xs, cache=self.decoder_cache)[0]
        xs = self.decoder.step(xs)                               # [B, 1, chunk_samples]
        return xs

    def warmup(self):
        """
        Warmup the codec by running a dummy encode/decode cycle.
        
        This compiles MLX kernels and allocates memory, reducing
        latency on the first real inference call. Essential for
        real-time applications where first-call latency matters.
        
        The warmup processes 4 frames (320ms) of silence to ensure
        all code paths are compiled.
        """
        pcm = mx.zeros((1, 1, 1920 * 4))  # 4 frames of silence
        codes = self.encode(pcm)
        pcm_out = self.decode(codes)
        mx.eval(pcm_out)  # Force evaluation

    @property
    def frame_rate(self) -> float:
        """
        Output frame rate in Hz.
        
        This is the rate at which tokens are produced/consumed.
        Typically 12.5 Hz, meaning one token frame per 80ms.
        
        Returns:
            Frame rate (typically 12.5)
        """
        return self.cfg.frame_rate

    @property
    def sample_rate(self) -> float:
        """
        Audio sample rate in Hz.
        
        This is the rate of the input/output audio waveforms.
        Typically 24000 Hz.
        
        Returns:
            Sample rate (typically 24000)
        """
        return self.cfg.sample_rate

    def load_pytorch_weights(
        self,
        file: str,
        strict: bool = True,
    ) -> nn.Module:
        """
        Load weights from a PyTorch checkpoint file.
        
        Handles the complex conversion from PyTorch weight naming conventions
        to MLX naming conventions. This includes:
        
        1. NAME REMAPPING
           - PyTorch uses different layer naming (e.g., "encoder.model.0")
           - MLX uses structured naming (e.g., "encoder.init_conv1d")
        
        2. WEIGHT TRANSPOSITION
           - PyTorch conv weights: [out_channels, in_channels, kernel_size]
           - MLX conv weights: [out_channels, kernel_size, in_channels]
           - ConvTranspose has different layout too
        
        3. SPECIAL HANDLING
           - EuclideanCodebook needs initialization flag update
           - ConvTranspose1d needs in-place weight update
        
        =======================================================================
        WEIGHT MAPPING DETAILS
        =======================================================================
        
        PyTorch → MLX name mappings:
        - "encoder.model.0" → "encoder.init_conv1d"
        - "encoder.model.14" → "encoder.final_conv1d"
        - "decoder.model.0" → "decoder.init_conv1d"
        - "decoder.model.14" → "decoder.final_conv1d"
        - Layer indices are remapped for encoder/decoder blocks
        
        Weight transpositions:
        - Conv weights: swapaxes(-1, -2)
        - ConvTranspose weights: transpose(1, 2, 0)
        
        =======================================================================
        
        Args:
            file: Path to PyTorch checkpoint (.safetensors format)
            strict: If True, raise error on missing/unexpected keys.
                   Set to False for partial loading.
        
        Returns:
            Self for method chaining
        
        Example:
            >>> mimi = Mimi(mimi_202407(8))
            >>> mimi.load_pytorch_weights("mimi.safetensors")
            >>> # Now ready for inference
        """
        weights = []
        for k, v in mx.load(file).items():
            v: mx.array = v
            # Remove leading underscores from path components
            k: str = ".".join([s.removeprefix("_") for s in k.split(".")])
            
            # =================================================================
            # ENCODER/DECODER MODEL PREFIX REMOVAL
            # =================================================================
            if k.startswith("encoder.model."):
                k = k.replace("encoder.model.", "encoder.")
            if k.startswith("decoder.model."):
                k = k.replace("decoder.model.", "decoder.")
            
            # =================================================================
            # ATTENTION PROJECTION RENAMING
            # =================================================================
            if k.endswith(".in_proj_weight"):
                k = k.replace(".in_proj_weight", ".in_proj.weight")
            
            # =================================================================
            # GATING LAYER RENAMING
            # =================================================================
            if k.endswith(".linear1.weight"):
                k = k.replace(".linear1.weight", ".gating.linear1.weight")
            if k.endswith(".linear2.weight"):
                k = k.replace(".linear2.weight", ".gating.linear2.weight")
            
            # =================================================================
            # DECODER LAYER INDEX REMAPPING
            # =================================================================
            # PyTorch decoder structure: [init, upsample+residual blocks, final]
            # Indices: 0, [2,3], [5,6], [8,9], [11,12], 14
            # MLX structure: init_conv1d, layers[0-3].{upsample,residuals}, final_conv1d
            for layerIdx, decoderIdx in enumerate([2, 5, 8, 11]):
                k = k.replace(
                    f"decoder.{decoderIdx}.", f"decoder.layers.{layerIdx}.upsample."
                )
                k = k.replace(
                    f"decoder.{decoderIdx + 1}.",
                    f"decoder.layers.{layerIdx}.residuals.0.",
                )
            
            # =================================================================
            # ENCODER LAYER INDEX REMAPPING
            # =================================================================
            # PyTorch encoder structure: [init, residual+downsample blocks, final]
            # Indices: 0, [1,3], [4,6], [7,9], [10,12], 14
            # MLX structure: init_conv1d, layers[0-3].{residuals,downsample}, final_conv1d
            for layerIdx, encoderIdx in enumerate([1, 4, 7, 10]):
                k = k.replace(
                    f"encoder.{encoderIdx}.", f"encoder.layers.{layerIdx}.residuals.0."
                )
                k = k.replace(
                    f"encoder.{encoderIdx + 2}.",
                    f"encoder.layers.{layerIdx}.downsample.",
                )

            # =================================================================
            # INIT/FINAL CONV RENAMING
            # =================================================================
            k = k.replace("decoder.0.", "decoder.init_conv1d.")
            k = k.replace("decoder.14.", "decoder.final_conv1d.")
            k = k.replace("encoder.0.", "encoder.init_conv1d.")
            k = k.replace("encoder.14.", "encoder.final_conv1d.")
            
            # =================================================================
            # RESIDUAL BLOCK INDEX REMAPPING
            # =================================================================
            k = k.replace(".block.1.", ".block.0.")
            k = k.replace(".block.3.", ".block.1.")

            # =================================================================
            # WEIGHT TRANSPOSITION FOR CONV LAYERS
            # =================================================================
            # PyTorch conv: [out_channels, in_channels, kernel_size]
            # MLX conv: [out_channels, kernel_size, in_channels]
            if (
                k.endswith(".conv.weight")
                or k.endswith(".output_proj.weight")
                or k.endswith(".input_proj.weight")
            ):
                v = v.swapaxes(-1, -2)
            
            # PyTorch conv-transpose: [in_channels, out_channels, kernel_size]
            # MLX conv-transpose: [out_channels, kernel_size, in_channels]
            if k.endswith(".convtr.weight"):
                v = v.transpose(1, 2, 0)
            
            weights.append((k, v))
        
        # Load the remapped weights
        m = self.load_weights(weights, strict=strict)

        # =================================================================
        # POST-LOAD UPDATES
        # =================================================================
        # Some modules need special handling after weight loading
        def _filter_fn(module, name, _):
            # EuclideanCodebook needs to mark itself as initialized
            if isinstance(module, EuclideanCodebook) and name == "initialized":
                module.update_in_place()
            # ConvTranspose1d needs to update its internal state
            if isinstance(module, ConvTranspose1d) and name == "weight":
                module.update_in_place()
            return True

        m.filter_and_map(_filter_fn)
        return m
