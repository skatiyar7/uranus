# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Language Model Implementation (Moshi Core)
==========================================

This module implements the core language model (Lm) for Moshi, a speech-text
foundation model for real-time, full-duplex dialogue. The architecture enables
sub-200ms latency speech-to-speech interaction.

=============================================================================
MOSHI ARCHITECTURE OVERVIEW (from the paper)
=============================================================================

Moshi uses a two-level autoregressive hierarchy called "RQ-Transformer":

1. TEMPORAL TRANSFORMER (Helium - 7B parameters)
   - A large decoder-only transformer initialized from a pure text LLM
   - Operates over TIME STEPS (12.5 Hz frame rate)
   - Processes SUMMED embeddings from all token streams
   - Produces a context vector z_s for each timestep
   - Does NOT directly predict any tokens - only provides hidden states

2. DEPTH TRANSFORMER (DepFormer - smaller, ~6 layers)
   - A smaller transformer that operates WITHIN a single timestep
   - Generates tokens autoregressively across codebook levels
   - Predicts: text tokens (Inner Monologue), semantic audio, acoustic audio
   - Conditioned on the Temporal Transformer's context vector

=============================================================================
TOKEN STREAMS AND EMBEDDING SUMMATION
=============================================================================

At each timestep t, Moshi models multiple parallel token streams:

For Moshi (system):
  - Text token (Inner Monologue) - guides speech generation
  - Semantic audio token (RVQ level 1) - linguistic content
  - Acoustic audio tokens (RVQ levels 2-8) - voice quality, prosody

For User:
  - Semantic audio token
  - Acoustic audio tokens

The CRITICAL operation is the embedding summation before the Temporal Transformer:

    x_t = Σ_k E^(k)(V_{t,k}) ∈ ℝ^d_model

Where:
  - E^(k) is the embedding table for token type k
  - V_{t,k} is the token value at timestep t for stream k
  - Each token type has its OWN embedding table (no sharing)

This summed embedding is what the Temporal Transformer (Helium) sees.

=============================================================================
DELAYED STREAMS ARCHITECTURE
=============================================================================

Audio tokens have DELAYS relative to the text stream:
  - Semantic tokens: typically delay=0 or small
  - Acoustic tokens: delay=1-2 frames

This means audio token at step T was generated based on context from step T-delay.

Benefits:
  - Temporal Transformer captures semantic-acoustic dependencies
  - Enables streaming generation with proper causal structure
  - Improves stability and intelligibility

=============================================================================
INNER MONOLOGUE
=============================================================================

Moshi predicts time-aligned text tokens as a PREFIX to its audio tokens.
This hierarchical decomposition: Text → Semantic → Acoustic

Benefits:
  - Stronger linguistic coherence
  - Improved factuality
  - Longer, more consistent speech generations
  - Enables streaming ASR/TTS from the same model

=============================================================================
KEY CLASSES
=============================================================================

- LmConfig: Configuration dataclass for the complete language model
- DepFormerConfig: Configuration for the Depth Transformer
- Lm: Main language model combining Temporal + Depth Transformers
- DepFormer: The Depth Transformer for audio token generation
- DepFormerSlice: Single slice generating one audio codebook level
- ScaledEmbedding: Embedding with optional low-rank factorization

=============================================================================
CONFIGURATION FUNCTIONS
=============================================================================

- config_v0_1(): Original Moshi 7B (d_model=4096, 32 layers)
- config1b_202412(): 1B parameter model (d_model=2048, 16 layers)
- config_helium_1_preview_2b(): Text-only Helium (no DepFormer)
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from ..modules.conditioner import (
    ConditionProvider,
    ConditionTensor,
    LutConditionerConfig,
    TensorConditionerConfig,
)
from ..modules.transformer import LayerCache, Transformer, TransformerConfig
from ..utils import sampling


@dataclass
class DepFormerConfig:
    """
    Configuration for the Depth Transformer (DepFormer).
    
    ==========================================================================
    ROLE IN MOSHI ARCHITECTURE
    ==========================================================================
    
    The DepFormer is the SECOND level of Moshi's two-level autoregressive
    hierarchy. While the Temporal Transformer (Helium) models dependencies
    across TIME, the DepFormer models dependencies WITHIN a single timestep
    across different token types (codebook levels).
    
    The DepFormer generates audio tokens in a specific order:
    1. First slice: Takes text token + Temporal Transformer output → predicts semantic audio
    2. Subsequent slices: Takes previous audio token → predicts next acoustic level
    
    This autoregressive structure across codebooks ensures proper dependencies:
    - Acoustic tokens depend on semantic tokens
    - Higher RVQ levels depend on lower levels
    
    ==========================================================================
    ARCHITECTURE DETAILS
    ==========================================================================
    
    - Typically 6 layers, d_model=1024, 16 heads
    - Much smaller than the Temporal Transformer (which is 32 layers, d_model=4096)
    - All slices SHARE the same transformer weights (with optional scheduling)
    - Each slice has its own embedding and output projection
    
    Attributes:
        transformer: Configuration for the shared transformer layers
        num_slices: Number of audio codebooks to generate (typically 8 for Moshi)
                   This equals the number of RVQ levels in the Mimi codec
        weights_per_step_schedule: Optional schedule for weight sharing across slices.
                                   If None, all slices share identical weights.
                                   If provided, maps slice_idx → weight_idx for selective sharing.
        low_rank_embeddings: Optional dimension for low-rank embedding factorization.
                            Reduces memory for large vocabularies by factorizing
                            the embedding matrix: E = E_low @ W_project
    """
    transformer: TransformerConfig
    num_slices: int
    weights_per_step_schedule: list[int] | None = None
    low_rank_embeddings: int | None = None


@dataclass
class LmConfig:
    """
    Configuration for the main Moshi Language Model.
    
    ==========================================================================
    COMPLETE MODEL CONFIGURATION
    ==========================================================================
    
    This dataclass holds ALL configuration parameters for the full Moshi model,
    encompassing both the Temporal Transformer (Helium) and the Depth Transformer
    (DepFormer), plus vocabulary sizes, audio settings, and conditioning options.
    
    ==========================================================================
    VOCABULARY STRUCTURE
    ==========================================================================
    
    TEXT VOCABULARY:
    - text_in_vocab_size: Input vocabulary (32001 for Moshi v0.1)
      Includes special tokens like padding, start-of-sequence
    - text_out_vocab_size: Output vocabulary (32000 for Moshi v0.1)
      The actual SentencePiece vocabulary size
    
    AUDIO VOCABULARY:
    - audio_vocab_size: Codebook size per RVQ level (2049 for Moshi)
      2048 actual codes + 1 padding token
    - audio_codebooks: Total number of audio codebooks (16 or 32 for full-duplex)
      = 8 or 16 (Moshi's speech) + 8 or 16 (User's speech)
    
    ==========================================================================
    DELAYED STREAMS
    ==========================================================================
    
    audio_delays: List of delays for each audio codebook relative to text.
    
    For Moshi v0.1: [0, 1, 1, 1, 1, 1, 1, 1] * 2 (for both speakers)
    For newer models: [0, 2, 2, 2, 2, 2, 2, 2] * 2 (for both speakers)
    - Semantic token (first): delay=0 (aligned with text)
    - Acoustic tokens (rest): delay=1 or 2 (one or two frames behind)
    
    This delay structure is CRITICAL for the model's ability to:
    1. Generate coherent audio from text guidance
    2. Maintain proper causal dependencies
    3. Enable streaming inference
    
    ==========================================================================
    MULTI-STREAM (FULL-DUPLEX) SUPPORT
    ==========================================================================
    
    demux_second_stream: When True, the model handles two audio streams
    (Moshi + User) multiplexed into single tokens. The ScaledEmbedding
    layer demultiplexes them using: combined = tok2 * card + tok1
    
    Attributes:
        transformer: Configuration for the Temporal Transformer (Helium)
        depformer: Configuration for the Depth Transformer
        text_in_vocab_size: Input text vocabulary size (includes special tokens)
        text_out_vocab_size: Output text vocabulary size (actual vocab)
        audio_vocab_size: Audio codebook vocabulary size (typically 2049)
        audio_codebooks: Total number of audio codebooks (input + output)
        audio_delays: Delay in steps for each audio codebook
        conditioners: Dictionary of conditioner configurations for guided generation
        demux_second_stream: Whether to demultiplex a second input stream
        extra_heads_num_heads: Number of extra prediction heads (auxiliary tasks)
        extra_heads_dim: Dimension of extra prediction heads
    """
    transformer: TransformerConfig
    depformer: DepFormerConfig
    text_in_vocab_size: int
    text_out_vocab_size: int
    audio_vocab_size: int
    audio_codebooks: int
    audio_delays: list[int]
    conditioners: dict[str, LutConditionerConfig | TensorConditionerConfig]
    demux_second_stream: bool = False
    extra_heads_num_heads: int = 0
    extra_heads_dim: int = 6

    @property
    def generated_codebooks(self):
        """
        Number of audio codebooks generated by the DepFormer.
        
        In full-duplex Moshi:
        - DepFormer generates 8 codebooks (Moshi's speech)
        - The other 8 codebooks come from user input (encoded by Mimi)
        
        Returns:
            int: Number of codebooks the model generates (typically 8)
        """
        if self.depformer is None:
            return 0
        return self.depformer.num_slices

    @property
    def other_codebooks(self):
        """
        Number of audio codebooks from input (not generated by DepFormer).
        
        These are the user's speech codebooks that are:
        1. Encoded from user audio by Mimi
        2. Fed as INPUT to the model (not predicted)
        3. Used to condition Moshi's responses
        
        In full-duplex: other_codebooks = 16 - 8 = 8 (user's speech)
        
        Returns:
            int: Number of input-only codebooks
        """
        return self.audio_codebooks - self.generated_codebooks

    @classmethod
    def from_config_dict(cls, data: dict) -> "LmConfig":
        """
        Create an LmConfig from a dictionary (typically loaded from JSON).
        
        This factory method handles the conversion from the JSON configuration
        format used in model checkpoints to the structured LmConfig dataclass.
        It's the primary way to instantiate LmConfig when loading pretrained models.
        
        =======================================================================
        JSON CONFIG STRUCTURE (example from moshi_7b_202409.json)
        =======================================================================
        
        {
            "dim": 4096,              # Temporal Transformer dimension
            "num_heads": 32,          # Attention heads
            "num_layers": 32,         # Transformer layers
            "depformer_dim": 1024,    # DepFormer dimension
            "depformer_num_heads": 16,
            "depformer_num_layers": 6,
            "dep_q": 8,               # Number of DepFormer slices
            "n_q": 16,                # Total audio codebooks
            "text_card": 32000,       # Text vocabulary size
            "card": 2048,             # Audio codebook size
            "delays": [0, 0, 1, ...], # Delays for each stream
            ...
        }
        
        Args:
            data: Dictionary containing model configuration from JSON
        
        Returns:
            LmConfig instance with all parameters properly set
        
        Note:
            The delays[0] is for text (always 0), so we skip it with [1:]
            for audio_delays since text delay is implicit.
        """
        # =====================================================================
        # TEMPORAL TRANSFORMER (HELIUM) CONFIGURATION
        # =====================================================================
        # This is the large backbone transformer that processes the summed
        # embeddings from all token streams across time.
        transformer = TransformerConfig(
            d_model=data["dim"],                    # Model dimension (4096 for 7B)
            num_heads=data["num_heads"],            # Attention heads (32 for 7B)
            num_layers=data["num_layers"],          # Layers (32 for 7B)
            dim_feedforward=4 * data["dim"],        # FFN dimension (4x model dim)
            causal=data["causal"],                  # Always True for autoregressive
            norm_first=True,                        # Pre-norm (RMSNorm before attention)
            bias_ff=False,                          # No bias in FFN (modern practice)
            bias_attn=False,                        # No bias in attention
            layer_scale=data["layer_scale"],        # Optional layer scaling
            context=data["context"],                # Context window (3000 frames = 4 min)
            max_period=data["max_period"],          # RoPE max period
            use_conv_block=False,                   # No conv blocks in main transformer
            use_conv_bias=True,
            cross_attention=data.get("cross_attention", False),  # For TTS conditioning
            gating=True,                            # GLU activation in FFN
            norm="rms_norm",                        # RMSNorm (not LayerNorm)
            positional_embedding=data["positional_embedding"],  # "rope"
            conv_layout=False,
            conv_kernel_size=3,
            kv_repeat=1,                            # No GQA by default
            max_seq_len=4096,
        )
        
        # =====================================================================
        # DEPTH TRANSFORMER (DEPFORMER) CONFIGURATION
        # =====================================================================
        # This smaller transformer generates audio tokens within each timestep,
        # conditioned on the Temporal Transformer's output.
        depformer = DepFormerConfig(
            transformer=TransformerConfig(
                d_model=data["depformer_dim"],              # Smaller dim (1024)
                num_heads=data["depformer_num_heads"],      # 16 heads
                num_layers=data["depformer_num_layers"],    # 6 layers
                dim_feedforward=data["depformer_dim_feedforward"],
                causal=data.get("depformer_causal", True),
                norm_first=True,
                bias_ff=False,
                bias_attn=data.get("depformer_layer_scale", False),
                layer_scale=None,
                context=data.get("depformer_context", data["dep_q"]),  # Context = num slices
                max_period=data.get("depformer_max_period", 8),
                use_conv_block=False,
                use_conv_bias=True,
                cross_attention=False,              # No cross-attention in DepFormer
                gating=True,
                norm="rms_norm",
                positional_embedding=data["depformer_pos_emb"],  # Often "none"
                conv_layout=False,
                conv_kernel_size=3,
                kv_repeat=1,
                max_seq_len=4096,
            ),
            num_slices=data["dep_q"],               # Number of audio codebooks to generate
            weights_per_step_schedule=data.get(
                "depformer_weights_per_step_schedule", None
            ),
            low_rank_embeddings=data.get("depformer_low_rank_embeddings", None),
        )
        
        # =====================================================================
        # CONDITIONERS CONFIGURATION
        # =====================================================================
        # Conditioners enable guided generation (e.g., speaker identity, quality)
        conditioners = {}
        if "conditioners" in data:
            for _name, _cfg in data["conditioners"].items():
                if _cfg["type"] == "lut":
                    # Lookup table conditioner (discrete conditioning)
                    _cfg = _cfg["lut"]
                    _cfg = LutConditionerConfig(
                        n_bins=_cfg["n_bins"],
                        dim=_cfg["dim"],
                        tokenizer=_cfg["tokenizer"],
                        possible_values=_cfg["possible_values"],
                    )
                elif _cfg["type"] == "tensor":
                    # Tensor conditioner (continuous conditioning, e.g., speaker embedding)
                    _cfg = _cfg["tensor"]
                    _cfg = TensorConditionerConfig(
                        dim=_cfg["dim"],
                    )
                else:
                    raise ValueError(f"unsupported conditioner type {_cfg['type']}")
                conditioners[_name] = _cfg
        
        return LmConfig(
            transformer=transformer,
            depformer=depformer,
            text_in_vocab_size=data["text_card"] + 1,   # +1 for special tokens
            text_out_vocab_size=data["text_card"],
            audio_vocab_size=data["card"] + 1,          # +1 for padding token
            audio_delays=data["delays"][1:],            # Skip text delay (always 0)
            audio_codebooks=data["n_q"],
            demux_second_stream=data.get("demux_second_stream", False),
            conditioners=conditioners,
            extra_heads_dim=data.get("extra_heads_dim", 6),
            extra_heads_num_heads=data.get("extra_heads_num_heads", 0),
        )

    @property
    def audio_eos_token(self) -> int:
        """
        End-of-sequence token for audio streams.
        
        Used to signal the end of audio generation. In the vocabulary layout:
        - Indices 0 to 2047: actual audio codes from Mimi codec
        - Index 2048: EOS token
        - Index 2049 (if present): padding token
        
        Returns:
            int: The EOS token index (vocab_size - 2)
        """
        return self.audio_vocab_size - 2

    @property
    def audio_padding_token(self) -> int:
        """
        Padding token for audio streams.
        
        Used to pad audio sequences to uniform length and to indicate
        "no audio" positions (e.g., before the delay window starts).
        
        Returns:
            int: The padding token index (vocab_size - 1, typically 2048)
        """
        return self.audio_vocab_size - 1


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer with optional low-rank factorization and stream demultiplexing.
    
    ==========================================================================
    ROLE IN MOSHI ARCHITECTURE
    ==========================================================================
    
    ScaledEmbedding is used for ALL token embeddings in Moshi:
    - Text embeddings (self.text_emb in Lm)
    - Audio embeddings for each codebook (self.audio_embs in Lm)
    - DepFormer input embeddings (slice.emb in DepFormerSlice)
    
    These embeddings are SUMMED together before the Temporal Transformer:
        x_t = text_emb(text_token) + Σ audio_emb[k](audio_token[k])
    
    ==========================================================================
    KEY FEATURES
    ==========================================================================
    
    1. ZERO INDEX (-1):
       A special index that produces ZERO embeddings. This is crucial for:
       - Indicating "no input" positions
       - Masking during classifier-free guidance
       - Handling positions before delay windows
    
    2. LOW-RANK FACTORIZATION:
       For large vocabularies, factorizes E ∈ ℝ^{V×D} into:
           E_low ∈ ℝ^{V×R} and W ∈ ℝ^{R×D}
       where R << D, reducing memory from O(V*D) to O(V*R + R*D)
    
    3. STREAM DEMULTIPLEXING (for full-duplex):
       When demux_second_stream=True, a single token encodes TWO streams:
           combined = tok2 * card + tok1
       
       The layer extracts both tokens and applies separate projections:
           output = out1(emb[tok1]) + out2(emb[tok2])
       
       This enables modeling both Moshi and User speech in a single sequence.
    
    Args:
        num_embeddings: Size of the vocabulary (e.g., 32001 for text, 2049 for audio)
        embedding_dim: Dimension of the embedding vectors (e.g., 4096)
        zero_idx: Special index that produces zero embeddings (must be negative, default -1)
        low_rank: If provided, use low-rank factorization with this inner dimension
        demux_second_stream: If True, demultiplex two streams from input tokens
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        zero_idx: int = -1,
        low_rank: int | None = None,
        demux_second_stream: bool = False,
    ):
        # Initialize base embedding with low_rank dimension if specified
        # Otherwise use full embedding_dim
        super().__init__(num_embeddings, low_rank or embedding_dim)
        
        # Zero index must be negative to avoid collision with valid token indices
        assert zero_idx < 0, "Please use negative values for the zero_idx."
        self.num_embeddings = num_embeddings
        self.zero_idx = zero_idx
        
        # Optional low-rank projection: E_low @ W_project
        self.low_rank = None
        if low_rank is not None:
            self.low_rank = nn.Linear(low_rank, embedding_dim, bias=False)

        # Stream demultiplexing for full-duplex operation
        self.demux_second_stream = demux_second_stream
        assert self.zero_idx == -1, (
            "When demuxing a second stream, zero_idx must be -1."
        )
        if self.demux_second_stream:
            # Separate output projections for each stream
            # This allows the model to learn different representations
            # for Moshi's speech vs User's speech
            self.out1 = nn.Linear(low_rank or embedding_dim, embedding_dim, bias=False)
            self.out2 = nn.Linear(low_rank or embedding_dim, embedding_dim, bias=False)

    def __call__(self, input: mx.array) -> mx.array:
        """
        Compute embeddings for input tokens.
        
        Handles three cases:
        1. Standard embedding lookup (no demux, no low-rank)
        2. Low-rank factorized embedding
        3. Demultiplexed two-stream embedding
        
        Args:
            input: Token indices [B, T] or [B, T, 1]
                   Can contain zero_idx (-1) for zero embeddings
        
        Returns:
            Embeddings [B, T, embedding_dim]
        """
        # =====================================================================
        # ZERO TOKEN HANDLING
        # =====================================================================
        # Identify positions with zero_idx (-1) that should produce zero embeddings
        is_zero = input == self.zero_idx
        zero = mx.zeros(1, dtype=input.dtype)
        
        # Clamp negative indices to 0 for valid embedding lookup
        # (we'll mask these positions to zero later)
        input = mx.maximum(input, 0)
        
        if self.demux_second_stream:
            # =================================================================
            # DEMULTIPLEXED TWO-STREAM EMBEDDING
            # =================================================================
            # Input encodes two tokens: combined = tok2 * card + tok1
            # This is used for full-duplex where we model both speakers
            
            # Extract the two multiplexed tokens
            left = input % self.num_embeddings      # tok1 (primary stream)
            right = input // self.num_embeddings    # tok2 + 1 (secondary stream)
            
            # Right stream uses -1 as zero value, so we subtract 1
            # This allows encoding -1 (zero) through the multiplexing
            right = right - 1
            
            # Look up embeddings for primary stream
            left = self.weight[left]
            
            # Handle zero values in secondary stream
            right_zero = (right < 0)[..., None]
            right = mx.maximum(right, 0)
            right = self.weight[right]
            
            # Combine with separate projections for each stream
            # out1 processes Moshi's tokens, out2 processes User's tokens
            y = self.out1(left) + mx.where(right_zero, zero, self.out2(right))
            
            # Apply zero mask for positions with zero_idx
            y = mx.where(is_zero[..., None], zero, y)
        else:
            # =================================================================
            # STANDARD EMBEDDING (with optional low-rank)
            # =================================================================
            # Simple embedding lookup
            y = self.weight[input]
            
            # Apply zero mask for positions with zero_idx
            y = mx.where(is_zero[..., None], zero, y)
            
            # Apply low-rank projection if configured
            if self.low_rank is not None:
                y = self.low_rank(y)
        
        return y


class DepFormerSlice(nn.Module):
    """
    A single slice of the DepFormer that generates one audio codebook level.
    
    ==========================================================================
    ROLE IN THE DEPTH TRANSFORMER
    ==========================================================================
    
    The DepFormer consists of multiple slices, each responsible for generating
    ONE level of the RVQ (Residual Vector Quantization) hierarchy:
    
    Slice 0: text_token → semantic_audio (RVQ level 1)
    Slice 1: semantic_audio → acoustic_audio_1 (RVQ level 2)
    Slice 2: acoustic_audio_1 → acoustic_audio_2 (RVQ level 3)
    ...
    Slice 7: acoustic_audio_6 → acoustic_audio_7 (RVQ level 8)
    
    This autoregressive structure across codebooks ensures:
    - Acoustic tokens properly depend on semantic content
    - Higher RVQ levels refine lower levels
    - Coherent audio generation
    
    ==========================================================================
    COMPUTATION FLOW
    ==========================================================================
    
    For each slice at timestep t:
    
    1. INPUT COMBINATION:
       input = linear_in(z_t) + emb(prev_token)
       
       Where:
       - z_t is the Temporal Transformer's context vector
       - prev_token is either text (slice 0) or previous audio (slice 1+)
    
    2. TRANSFORMER PROCESSING:
       hidden = transformer(input, cache=shared_cache)
       
       Note: All slices SHARE the same transformer weights!
    
    3. OUTPUT PROJECTION:
       logits = linear_out(hidden) → ℝ^{audio_vocab_size - 1}
       
       The -1 is because we don't predict the padding token.
    
    ==========================================================================
    WEIGHT SHARING
    ==========================================================================
    
    All slices share the same transformer weights, but have SEPARATE:
    - Input embeddings (emb): Different vocab sizes (text vs audio)
    - Input projections (linear_in): Same dim but separate weights
    - Output projections (linear_out): Separate for each codebook
    
    This weight sharing is efficient and works because the transformer
    learns a general "codebook prediction" function.
    
    Args:
        in_vocab_size: Input vocabulary size
                      - Slice 0: text_in_vocab_size (32001)
                      - Slice 1+: audio_vocab_size (2049)
        out_vocab_size: Output vocabulary size (audio_vocab_size - 1 = 2048)
        main_transformer_dim: Dimension of the Temporal Transformer output (4096)
        demux_second_stream: Whether to demultiplex two streams (for slice 0 only)
        cfg: DepFormer configuration
    """

    def __init__(
        self,
        in_vocab_size: int,
        out_vocab_size: int,
        main_transformer_dim: int,
        demux_second_stream: bool,
        cfg: DepFormerConfig,
    ):
        super().__init__()

        dim = cfg.transformer.d_model  # DepFormer dimension (1024)
        
        # Input embedding for the previous token (text or audio)
        self.emb = ScaledEmbedding(
            in_vocab_size,
            dim,
            low_rank=cfg.low_rank_embeddings,
            demux_second_stream=demux_second_stream,
        )
        
        # Project Temporal Transformer output to DepFormer dimension
        # 4096 → 1024 (no bias for efficiency)
        self.linear_in = nn.Linear(main_transformer_dim, dim, bias=False)
        
        # Output projection to audio vocabulary
        # 1024 → 2048 (predicting audio codes, not padding)
        self.linear_out = nn.Linear(dim, out_vocab_size, bias=False)
        
        # The shared transformer (weights are actually shared across slices
        # through the weight loading mechanism, not here)
        self.transformer = Transformer(cfg.transformer)

    def __call__(self, _: mx.array) -> mx.array:
        """Not implemented - use DepFormer.sample() instead."""
        raise ValueError("not implemented")


class DepFormer(nn.Module):
    """
    Depth Transformer for autoregressive audio token generation.
    
    ==========================================================================
    ROLE IN MOSHI'S TWO-LEVEL HIERARCHY
    ==========================================================================
    
    The DepFormer is the SECOND level of Moshi's RQ-Transformer architecture:
    
    Level 1 (Temporal): Helium processes time steps, produces context z_t
    Level 2 (Depth):    DepFormer generates tokens WITHIN timestep t
    
    This two-level design avoids the need to flatten all audio tokens into
    one long sequence (which would be 8x longer and much slower).
    
    ==========================================================================
    GENERATION PROCESS
    ==========================================================================
    
    At each timestep t, given context z_t from the Temporal Transformer:
    
    1. Slice 0: z_t + emb(text_token) → transformer → sample semantic_audio
    2. Slice 1: z_t + emb(semantic_audio) → transformer → sample acoustic_1
    3. Slice 2: z_t + emb(acoustic_1) → transformer → sample acoustic_2
    ...
    8. Slice 7: z_t + emb(acoustic_6) → transformer → sample acoustic_7
    
    Key insight: The SAME z_t is used for all slices, but each slice
    conditions on the PREVIOUS slice's output.
    
    ==========================================================================
    WEIGHT SHARING
    ==========================================================================
    
    All slices share transformer weights but have separate:
    - Embeddings (different input vocab sizes)
    - Output projections (separate per codebook)
    
    Optional: weights_per_step_schedule allows selective weight sharing
    where some slices can share weights while others don't.
    
    ==========================================================================
    CLASSIFIER-FREE GUIDANCE (CFG)
    ==========================================================================
    
    When cfg_coef != 1.0, the DepFormer computes:
        logits = cfg_coef * conditional - (cfg_coef - 1) * unconditional
    
    This improves generation quality by amplifying the model's confidence
    in its predictions while suppressing uncertain outputs.
    
    Args:
        cfg: LmConfig containing depformer configuration
    """

    def __init__(self, cfg: LmConfig):
        super().__init__()

        # Create one slice per audio codebook to generate
        self.slices: list[DepFormerSlice] = []
        for slice_idx in range(cfg.depformer.num_slices):
            # Slice 0 takes text input, others take audio input
            in_vs = cfg.text_in_vocab_size if slice_idx == 0 else cfg.audio_vocab_size
            
            slice = DepFormerSlice(
                in_vs,
                cfg.audio_vocab_size - 1,  # Don't predict padding token
                main_transformer_dim=cfg.transformer.d_model,
                # Only slice 0 needs demux (for text stream)
                demux_second_stream=slice_idx == 0 and cfg.demux_second_stream,
                cfg=cfg.depformer,
            )
            self.slices.append(slice)

    def __call__(self, _: mx.array) -> mx.array:
        """Not implemented - use sample() instead."""
        raise ValueError("not implemented")

    def sample(
        self,
        main_transformer_out: mx.array,
        sampler: sampling.Sampler,
        text_token: mx.array,
        cache: list[LayerCache],
        cfg_coef: float = 1.0,
    ) -> mx.array:
        """
        Sample audio tokens from all DepFormer slices autoregressively.
        
        This is the core audio generation loop that produces all 8 codebook
        levels for a single timestep, conditioned on the Temporal Transformer's
        output and the predicted text token.
        
        =======================================================================
        ALGORITHM
        =======================================================================
        
        tokens = []
        last_token = text_token
        
        for slice in slices:
            # Combine context and previous token
            input = slice.linear_in(z_t) + slice.emb(last_token)
            
            # Process through shared transformer
            hidden = slice.transformer(input, cache)
            
            # Predict next codebook level
            logits = slice.linear_out(hidden)
            
            # Apply CFG if enabled
            if cfg_coef != 1:
                logits = cfg_coef * cond - (cfg_coef - 1) * uncond
            
            # Sample token
            last_token = sampler(logits)
            tokens.append(last_token)
        
        return stack(tokens)
        
        =======================================================================
        
        Args:
            main_transformer_out: Context from Temporal Transformer [B, 1, D]
                                 This is z_t, the hidden state at timestep t
            sampler: Sampler for token selection (temperature, top-k, etc.)
            text_token: Text token to condition the first slice [B, 1]
                       This is the Inner Monologue text prediction
            cache: KV cache for the transformer (shared across slices)
                  Reset at the start of each timestep
            cfg_coef: Classifier-free guidance coefficient
                     1.0 = no guidance, >1.0 = stronger guidance
        
        Returns:
            Audio tokens for all codebooks [B, num_slices]
            Shape: [batch_size, 8] for standard Moshi
        """
        tokens = []
        last_token = text_token
        
        # =================================================================
        # RESET CACHE FOR NEW TIMESTEP
        # =================================================================
        # The cache is shared between slices but NOT persisted between
        # timesteps. Each timestep starts fresh.
        for c in cache:
            c.reset()
        
        # =================================================================
        # AUTOREGRESSIVE GENERATION ACROSS CODEBOOKS
        # =================================================================
        for slice in self.slices:
            # Note: The 2048 tokens should be teacher forced on the first
            # slices. However as delays are non-decreasing in the number
            # of slices, this is actually not necessary as the generated
            # tokens will end up not being used.

            # Handle CFG by duplicating inputs for conditional/unconditional
            if cfg_coef != 1:
                last_token = mx.tile(last_token, (2, 1))
            
            # ---------------------------------------------------------
            # COMBINE CONTEXT AND PREVIOUS TOKEN
            # ---------------------------------------------------------
            # This is the key operation: z_t provides global context,
            # while emb(last_token) provides local conditioning
            xs = slice.linear_in(main_transformer_out) + slice.emb(last_token)
            
            # ---------------------------------------------------------
            # PROCESS THROUGH TRANSFORMER
            # ---------------------------------------------------------
            # The transformer refines the combined representation
            xs = slice.transformer(xs, cache=cache)
            
            # ---------------------------------------------------------
            # PROJECT TO VOCABULARY
            # ---------------------------------------------------------
            logits = slice.linear_out(xs)
            
            # ---------------------------------------------------------
            # APPLY CLASSIFIER-FREE GUIDANCE
            # ---------------------------------------------------------
            if cfg_coef != 1:
                l1, l2 = logits.split(2, axis=0)
                # Amplify conditional, suppress unconditional
                logits = cfg_coef * l1 - (cfg_coef - 1) * l2

            # ---------------------------------------------------------
            # SAMPLE TOKEN
            # ---------------------------------------------------------
            last_token, _ = sampler(logits)
            tokens.append(last_token)
        
        # Stack all codebook tokens: [B, num_slices]
        tokens = mx.stack(tokens, axis=1)
        return tokens


class Lm(nn.Module):
    """
    Main Language Model for Moshi - The Complete RQ-Transformer.
    
    ==========================================================================
    ARCHITECTURE OVERVIEW
    ==========================================================================
    
    Lm is the complete Moshi model combining:
    
    1. TEMPORAL TRANSFORMER (self.transformer)
       - Initialized from Helium (7B text LLM)
       - Processes SUMMED embeddings across time
       - Produces context vectors z_t for each timestep
       - 32 layers, d_model=4096, 32 heads (for 7B version)
    
    2. DEPTH TRANSFORMER (self.depformer)
       - Generates audio tokens within each timestep
       - 8 slices for 8 RVQ codebook levels
       - 6 layers, d_model=1024, 16 heads
    
    3. EMBEDDINGS
       - self.text_emb: Text token embeddings
       - self.audio_embs: List of 16 audio embeddings (8 Moshi + 8 User)
    
    4. OUTPUT HEADS
       - self.text_linear: Text prediction head
       - self.extra_heads: Optional auxiliary prediction heads
    
    ==========================================================================
    THE CRITICAL EMBEDDING SUMMATION
    ==========================================================================
    
    At each timestep t, the input to the Temporal Transformer is:
    
        x_t = text_emb(text_t) + Σ_k audio_emb[k](audio_t,k) + condition
    
    This summation is performed in _sample():
        xs = self.text_emb(text_token_ids)
        for token_ids, emb in zip(audio_token_ids, self.audio_embs):
            xs = xs + emb(token_ids)
    
    The Temporal Transformer sees ONLY this sum, not individual tokens.
    This is analogous to how BERT sums token + position + segment embeddings.
    
    ==========================================================================
    GENERATION FLOW
    ==========================================================================
    
    For each timestep t:
    
    1. EMBEDDING SUMMATION
       x_t = text_emb + Σ audio_embs + condition
    
    2. TEMPORAL TRANSFORMER
       z_t = transformer(x_t, cache)  # Context vector
       z_t = out_norm(z_t)            # RMSNorm
    
    3. TEXT PREDICTION
       text_logits = text_linear(z_t)
       text_token = sample(text_logits)
    
    4. AUDIO GENERATION (via DepFormer)
       audio_tokens = depformer.sample(z_t, text_token)
    
    ==========================================================================
    FULL-DUPLEX OPERATION
    ==========================================================================
    
    In full-duplex mode (demux_second_stream=True):
    - audio_codebooks = 16 or 32 (8 or 16 Moshi + 8 or 16 User)
    - User's codebooks come from Mimi-encoded input audio
    - Moshi's codebooks are generated by the DepFormer
    - Both streams are modeled jointly for natural conversation
    
    ==========================================================================
    CONDITIONING
    ==========================================================================
    
    Optional conditioning via self.condition_provider:
    - LUT conditioners: Discrete conditioning (quality level, etc.)
    - Tensor conditioners: Continuous conditioning (speaker embedding)
    
    Conditioning is added to the embedding sum before the transformer.
    
    Args:
        cfg: LmConfig containing all model configuration
    """

    def __init__(self, cfg: LmConfig):
        super().__init__()

        dim = cfg.transformer.d_model  # 4096 for 7B model
        
        # =====================================================================
        # TEMPORAL TRANSFORMER (HELIUM)
        # =====================================================================
        # This is the large backbone transformer that processes the summed
        # embeddings from all token streams. Initialized from pretrained Helium.
        self.transformer: Transformer = Transformer(cfg.transformer)
        
        # =====================================================================
        # DEPTH TRANSFORMER (DEPFORMER)
        # =====================================================================
        # Generates audio tokens within each timestep, conditioned on the
        # Temporal Transformer's output.
        self.depformer: DepFormer = DepFormer(cfg)
        
        # =====================================================================
        # TEXT EMBEDDING
        # =====================================================================
        # Embeds text tokens (Inner Monologue) into the model dimension.
        # With demux_second_stream, can handle multiplexed two-speaker text.
        self.text_emb = ScaledEmbedding(
            cfg.text_in_vocab_size, dim, demux_second_stream=cfg.demux_second_stream
        )
        self.cfg: LmConfig = cfg

        # =====================================================================
        # OUTPUT NORMALIZATION
        # =====================================================================
        # Applied to transformer output before text prediction.
        # RMSNorm is more efficient than LayerNorm and works well for LLMs.
        if cfg.transformer.norm == "layer_norm":
            self.out_norm = nn.LayerNorm(dim, 1e-5)
        elif cfg.transformer.norm == "rms_norm":
            self.out_norm = nn.RMSNorm(dim, 1e-8)
        else:
            raise ValueError(f"unsupported norm type {cfg.transformer.norm}")

        # =====================================================================
        # TEXT OUTPUT HEAD
        # =====================================================================
        # Projects transformer output to text vocabulary for prediction.
        # This is the Inner Monologue prediction head.
        self.text_linear = nn.Linear(dim, cfg.text_out_vocab_size, bias=False)
        
        # =====================================================================
        # AUDIO EMBEDDINGS
        # =====================================================================
        # One embedding table per audio codebook.
        # For full-duplex: 16 embeddings (8 Moshi + 8 User)
        # Each has vocab_size=2049 (2048 codes + padding)
        self.audio_embs = [
            ScaledEmbedding(cfg.audio_vocab_size, dim)
            for _ in range(cfg.audio_codebooks)
        ]
        
        # =====================================================================
        # EXTRA PREDICTION HEADS (OPTIONAL)
        # =====================================================================
        # For auxiliary tasks like voice activity detection, emotion, etc.
        self.extra_heads = [
            nn.Linear(dim, cfg.extra_heads_dim, bias=False)
            for _ in range(cfg.extra_heads_num_heads)
        ]
        
        # =====================================================================
        # KV CACHES
        # =====================================================================
        # Caches for efficient autoregressive generation.
        # transformer_cache: For the Temporal Transformer (persists across steps)
        # depformer_cache: For the DepFormer (reset each step)
        self.transformer_cache: list[LayerCache] = self.transformer.make_rot_cache()

        if len(self.depformer.slices) > 0:
            self.depformer_cache: list[LayerCache] = self.depformer.slices[
                0
            ].transformer.make_cache()
        else:
            self.depformer_cache = []

        # =====================================================================
        # CONDITION PROVIDER (OPTIONAL)
        # =====================================================================
        # Handles conditioning for guided generation (speaker, quality, etc.)
        if len(cfg.conditioners) > 0:
            self.condition_provider = ConditionProvider(
                cfg.transformer.d_model, cfg.conditioners
            )
        else:
            self.condition_provider = None

    def load_pytorch_weights(
        self,
        file: str,
        lm_config: LmConfig,
        strict: bool = True,
    ) -> nn.Module:
        """
        Load weights from a PyTorch checkpoint into this MLX model.
        
        This method handles the complex weight mapping between PyTorch and MLX
        naming conventions. The PyTorch Moshi model has a different structure
        than the MLX implementation, particularly for:
        
        1. NORMALIZATION LAYERS
           PyTorch: "out_norm.alpha" (shape [1, 1, dim])
           MLX: "out_norm.weight" (shape [dim])
        
        2. AUDIO EMBEDDINGS
           PyTorch: "emb.{idx}.weight"
           MLX: "audio_embs.{idx}.weight"
        
        3. DEPFORMER STRUCTURE
           PyTorch has flat structure: "depformer_in.{idx}", "linears.{idx}"
           MLX has nested structure: "depformer.slices.{idx}.linear_in"
        
        4. WEIGHT SHARING IN DEPFORMER
           PyTorch shares attention weights across slices (split at load time)
           MLX has separate weights per slice
        
        Args:
            file: Path to PyTorch checkpoint (.safetensors format)
            lm_config: Configuration to determine weight mapping
            strict: If True, raise error on missing/unexpected keys
        
        Returns:
            Self for method chaining
        """
        # Load PyTorch weights
        pth_t = mx.load(file)
        
        # Determine number of weight chunks for DepFormer
        # (for weight sharing schedule)
        depformer_chunks = lm_config.depformer.num_slices
        if lm_config.depformer.weights_per_step_schedule is not None:
            depformer_chunks = max(lm_config.depformer.weights_per_step_schedule) + 1

        mlx_t = {}
        
        # =================================================================
        # OUTPUT NORMALIZATION
        # =================================================================
        # PyTorch uses "alpha" with shape [1, 1, dim], MLX uses "weight" with [dim]
        mlx_t["out_norm.weight"] = pth_t["out_norm.alpha"][0, 0]
        
        # =================================================================
        # TEXT EMBEDDINGS AND OUTPUT
        # =================================================================
        for name in [
            "text_emb.out1.weight",   # Demux projection 1
            "text_emb.out2.weight",   # Demux projection 2
            "text_emb.weight",        # Main embedding
            "text_linear.weight",     # Output projection
        ]:
            if name in pth_t:
                mlx_t[name] = pth_t[name]
        
        # =================================================================
        # AUDIO EMBEDDINGS
        # =================================================================
        # Map from PyTorch "emb.{idx}" to MLX "audio_embs.{idx}"
        for cb_idx in range(lm_config.audio_codebooks):
            mlx_t[f"audio_embs.{cb_idx}.weight"] = pth_t[f"emb.{cb_idx}.weight"]
        
        # =================================================================
        # TEMPORAL TRANSFORMER (HELIUM)
        # =================================================================
        for k, v in sorted(pth_t.items()):
            if k.startswith("transformer"):
                # Handle normalization alpha → weight conversion
                if k.endswith(".alpha"):
                    v = v[0, 0]
                k = k.replace(".alpha", ".weight")
                k = k.replace(".in_proj_weight", ".in_proj.weight")
                mlx_t[k] = v
            # Copy condition provider and extra heads directly
            if k.startswith("condition_provider.") or k.startswith("extra_heads."):
                mlx_t[k] = v

        # =================================================================
        # DEPFORMER WEIGHTS
        # =================================================================
        # The DepFormer has complex weight sharing that must be handled
        for slice_idx in range(lm_config.depformer.num_slices):
            # Determine which PyTorch weight index to use
            # (for weight sharing schedule)
            pth_idx = slice_idx
            if lm_config.depformer.weights_per_step_schedule is not None:
                pth_idx = lm_config.depformer.weights_per_step_schedule[slice_idx]
            
            slice_p = f"depformer.slices.{slice_idx}"
            
            # Input projection from Temporal Transformer
            mlx_t[f"{slice_p}.linear_in.weight"] = pth_t[
                f"depformer_in.{pth_idx}.weight"
            ]
            # Output projection to audio vocabulary
            mlx_t[f"{slice_p}.linear_out.weight"] = pth_t[f"linears.{slice_idx}.weight"]
            
            # Slice 0 uses text embedding, others use audio embedding
            if slice_idx == 0:
                mlx_t[f"{slice_p}.emb.weight"] = pth_t["depformer_text_emb.weight"]
                # Handle optional low-rank and demux projections
                for _n in ["low_rank", "out1", "out2"]:
                    if f"depformer_text_emb.{_n}.weight" in pth_t:
                        mlx_t[f"{slice_p}.emb.{_n}.weight"] = pth_t[
                            f"depformer_text_emb.{_n}.weight"
                        ]
            else:
                mlx_t[f"{slice_p}.emb.weight"] = pth_t[
                    f"depformer_emb.{slice_idx - 1}.weight"
                ]
                if f"depformer_emb.{slice_idx - 1}.low_rank.weight" in pth_t:
                    mlx_t[f"{slice_p}.emb.low_rank.weight"] = pth_t[
                        f"depformer_emb.{slice_idx - 1}.low_rank.weight"
                    ]
            
            # DepFormer transformer layers
            # Note: Attention weights are SHARED in PyTorch and SPLIT here
            for layer_idx in range(lm_config.depformer.transformer.num_layers):
                p = f"{slice_p}.transformer.layers.{layer_idx}"
                
                # Normalization layers
                mlx_t[f"{p}.norm1.weight"] = pth_t[
                    f"depformer.layers.{layer_idx}.norm1.alpha"
                ][0, 0]
                mlx_t[f"{p}.norm2.weight"] = pth_t[
                    f"depformer.layers.{layer_idx}.norm2.alpha"
                ][0, 0]
                
                # Gating (GLU) layers - indexed by pth_idx for weight sharing
                mlx_t[f"{p}.gating.linear_in.weight"] = pth_t[
                    f"depformer.layers.{layer_idx}.gating.{pth_idx}.linear_in.weight"
                ]
                mlx_t[f"{p}.gating.linear_out.weight"] = pth_t[
                    f"depformer.layers.{layer_idx}.gating.{pth_idx}.linear_out.weight"
                ]
                
                # Attention weights - SPLIT from shared PyTorch weights
                mlx_t[f"{p}.self_attn.in_proj.weight"] = mx.split(
                    pth_t[f"depformer.layers.{layer_idx}.self_attn.in_proj_weight"],
                    depformer_chunks,
                )[pth_idx]
                mlx_t[f"{p}.self_attn.out_proj.weight"] = mx.split(
                    pth_t[f"depformer.layers.{layer_idx}.self_attn.out_proj.weight"],
                    depformer_chunks,
                )[pth_idx]
        
        return self.load_weights(list(mlx_t.items()), strict=strict)

    @property
    def n_q(self) -> int:
        """
        Total number of audio codebooks (Moshi + User).
        
        For full-duplex Moshi: 16 (8 + 8)
        """
        return self.cfg.audio_codebooks

    @property
    def dep_q(self) -> int:
        """
        Number of codebooks generated by the DepFormer.
        
        For Moshi: 8 (Moshi's speech only, not User's)
        """
        return self.cfg.depformer.num_slices

    @property
    def audio_offset(self) -> int:
        """
        Offset for audio codebooks in the token sequence.
        
        Returns 1 because index 0 is text, indices 1+ are audio.
        """
        return 1

    @property
    def delays(self) -> list[int]:
        """
        Delay values for each audio codebook.
        
        Delays determine how many timesteps behind each codebook is
        relative to the text stream. This enables the delayed streams
        architecture that is key to Moshi's design.
        """
        return self.cfg.audio_delays

    def forward_text(
        self,
        token_ids: mx.array,
        cross_attention_src: None | mx.array = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Forward pass for text-only input (no audio).
        
        Used for text-only models like Helium or for text-only inference.
        Does NOT apply the DepFormer.
        
        Args:
            token_ids: Text token indices [B, T]
            cross_attention_src: Optional cross-attention source
        
        Returns:
            Tuple of (transformer_output, text_logits)
        """
        xs = self.text_emb(token_ids)
        transformer_out = self.transformer(
            xs, cache=self.transformer_cache, cross_attention_src=cross_attention_src
        )
        transformer_out = self.out_norm(transformer_out)
        text_logits = self.text_linear(transformer_out)
        return (transformer_out, text_logits)

    def __call__(
        self,
        token_ids: mx.array,
        cross_attention_src: None | mx.array = None,
    ) -> mx.array:
        """
        Forward pass returning only text logits.
        
        Convenience method for text-only inference.
        Does NOT apply the DepFormer.
        
        Args:
            token_ids: Text token indices [B, T]
            cross_attention_src: Optional cross-attention source
        
        Returns:
            Text logits [B, T, vocab_size]
        """
        xs = self.text_emb(token_ids)
        transformer_out = self.transformer(
            xs, cache=self.transformer_cache, cross_attention_src=cross_attention_src
        )
        transformer_out = self.out_norm(transformer_out)
        text_logits = self.text_linear(transformer_out)
        return text_logits

    def _sample(
        self,
        text_token_ids: mx.array,
        audio_token_ids: list[mx.array],
        text_sampler: sampling.Sampler,
        audio_sampler: sampling.Sampler,
        ct: ConditionTensor | None = None,
        cross_attention_src: None | mx.array = None,
        cfg_coef: float = 1.0,
        on_text_hook=None,
        on_audio_hook=None,
    ) -> tuple[mx.array, mx.array | None, mx.array]:
        """
        Core generation step: produce text and audio tokens for one timestep.
        
        =====================================================================
        THIS IS THE HEART OF MOSHI'S GENERATION
        =====================================================================
        
        This method implements the complete forward pass for one timestep:
        
        1. EMBEDDING SUMMATION (the critical operation from the paper)
           x_t = Σ_k E^(k)(V_{t,k}) ∈ ℝ^{d_model}
        
        2. TEMPORAL TRANSFORMER (Helium)
           z_t = transformer(x_t) → context vector
        
        3. TEXT PREDICTION (Inner Monologue)
           text_logits = text_linear(z_t)
           text_token = sample(text_logits)
        
        4. AUDIO GENERATION (DepFormer)
           audio_tokens = depformer.sample(z_t, text_token)
        
        =====================================================================
        EMBEDDING SUMMATION DETAILS
        =====================================================================
        
        The summation combines:
        - text_emb(text_token): Inner Monologue text embedding
        - audio_emb[0](audio[0]): Moshi semantic audio
        - audio_emb[1](audio[1]): Moshi acoustic level 1
        - ...
        - audio_emb[7](audio[7]): Moshi acoustic level 7
        - audio_emb[8](audio[8]): User semantic audio
        - ...
        - audio_emb[15](audio[15]): User acoustic level 7
        - ct.tensor: Optional conditioning (speaker, quality)
        
        The Temporal Transformer sees ONLY this sum, not individual tokens.
        
        =====================================================================
        CLASSIFIER-FREE GUIDANCE (CFG)
        =====================================================================
        
        When cfg_coef != 1.0:
        - Input is duplicated: [conditional, unconditional]
        - Both are processed through the transformer
        - Logits are interpolated: cfg_coef * cond - (cfg_coef - 1) * uncond
        
        This amplifies the model's confidence in its predictions.
        
        Args:
            text_token_ids: Previous text token [B, 1]
            audio_token_ids: List of previous audio tokens, one per codebook
                            Each is [1, B] (note: transposed for efficiency)
            text_sampler: Sampler for text token selection
            audio_sampler: Sampler for audio token selection
            ct: Optional condition tensor for guided generation
            cross_attention_src: Optional cross-attention source (for TTS)
            cfg_coef: Classifier-free guidance coefficient (1.0 = no guidance)
            on_text_hook: Callback after text token generation
            on_audio_hook: Callback after audio token generation
        
        Returns:
            Tuple of:
            - text_token: Generated text token [B, 1]
            - audio_tokens: Generated audio tokens [B, num_slices] or None
            - transformer_out: Transformer hidden state [B, 1, D]
        """
        # =====================================================================
        # STEP 1: EMBEDDING SUMMATION
        # =====================================================================
        # This is the critical operation: x_t = Σ_k E^(k)(V_{t,k})
        # Start with text embedding
        xs = self.text_emb(text_token_ids)
        
        # Add audio embeddings for all codebooks (Moshi + User)
        for token_ids, emb in zip(audio_token_ids, self.audio_embs):
            _emb = emb(token_ids)
            # Transpose from [1, B, D] to [B, 1, D] for proper broadcasting
            _emb = _emb.transpose(1, 0, 2)
            xs = xs + _emb
        
        # Add conditioning if provided (speaker embedding, quality, etc.)
        if ct is not None:
            xs = xs + mx.expand_dims(ct.tensor, axis=1)
        
        # For CFG: duplicate input for conditional and unconditional paths
        if cfg_coef != 1:
            xs = mx.tile(xs, (2, 1, 1))
        
        # =====================================================================
        # STEP 2: TEMPORAL TRANSFORMER (HELIUM)
        # =====================================================================
        # Process the summed embedding through the large transformer
        # This produces the context vector z_t that captures all temporal context
        transformer_out = self.transformer(
            xs,
            cache=self.transformer_cache,
            cross_attention_src=cross_attention_src,
        )
        
        # Apply output normalization (RMSNorm)
        transformer_out = self.out_norm(transformer_out)
        
        # =====================================================================
        # STEP 3: TEXT PREDICTION (INNER MONOLOGUE)
        # =====================================================================
        # Project to text vocabulary and sample
        text_logits = self.text_linear(transformer_out)
        
        # Apply CFG to text logits
        if cfg_coef != 1:
            l1, l2 = text_logits.split(2, axis=0)
            text_logits = cfg_coef * l1 - (cfg_coef - 1) * l2
        
        # Sample text token
        text_token, _ = text_sampler(text_logits)
        
        # Call text hook if provided (for TTS state machine, logging, etc.)
        if on_text_hook is not None:
            on_text_hook(text_token)
        
        # =====================================================================
        # STEP 4: AUDIO GENERATION (DEPFORMER)
        # =====================================================================
        # Generate audio tokens conditioned on transformer output and text token
        if len(self.depformer.slices) > 0:
            audio_tokens = self.depformer.sample(
                transformer_out,
                audio_sampler,
                text_token,
                self.depformer_cache,
                cfg_coef=cfg_coef,
            )
            # Call audio hook if provided
            if on_audio_hook is not None:
                on_audio_hook(audio_tokens)
        else:
            # No DepFormer (text-only model like Helium)
            audio_tokens = None
        
        return text_token, audio_tokens, transformer_out

    def sample(
        self,
        text_token_ids: mx.array,
        audio_token_ids: list[mx.array],
        text_sampler: sampling.Sampler,
        audio_sampler: sampling.Sampler,
        ct: ConditionTensor | None = None,
        cross_attention_src: None | mx.array = None,
        cfg_coef: float = 1.0,
        on_text_hook=None,
        on_audio_hook=None,
    ) -> tuple[mx.array, mx.array | None]:
        """
        Public interface for one generation step.
        
        Wrapper around _sample() that returns only the generated tokens,
        not the transformer hidden state.
        
        Args:
            text_token_ids: Previous text token [B, 1]
            audio_token_ids: List of previous audio tokens
            text_sampler: Sampler for text tokens
            audio_sampler: Sampler for audio tokens
            ct: Optional condition tensor
            cross_attention_src: Optional cross-attention source
            cfg_coef: Classifier-free guidance coefficient
            on_text_hook: Callback after text generation
            on_audio_hook: Callback after audio generation
        
        Returns:
            Tuple of (text_token, audio_tokens)
        """
        text, audio, _ = self._sample(
            text_token_ids,
            audio_token_ids,
            text_sampler,
            audio_sampler,
            ct,
            cross_attention_src,
            cfg_coef,
            on_text_hook,
            on_audio_hook,
        )
        return text, audio

    def warmup(self, ct: ConditionTensor | None = None):
        """
        Warmup the model by running a dummy generation step.
        
        This compiles MLX kernels and allocates memory, reducing latency
        on the first real inference call. Essential for real-time operation.
        
        The dummy check (sum == 42) ensures the computation actually runs
        and isn't optimized away by the compiler.
        
        Args:
            ct: Optional condition tensor to include in warmup
        """
        text, audio = self.sample(
            mx.array([[self.cfg.text_out_vocab_size]]),  # Start token
            [mx.array([[0]])] * self.cfg.other_codebooks,  # Dummy audio
            text_sampler=sampling.Sampler(),
            audio_sampler=sampling.Sampler(),
            ct=ct,
        )
        # Force evaluation with dummy check
        if text.sum().item() == 42:
            raise ValueError(42)
        if audio is not None and audio.sum().item() == 42:
            raise ValueError(42)
        # Reset cache after warmup
        for c in self.transformer_cache:
            c.reset()

# =============================================================================
# MODEL CONFIGURATION FUNCTIONS
# =============================================================================
# These functions create pre-defined configurations for different Moshi variants.
# Each configuration specifies the complete architecture including:
# - Temporal Transformer (Helium) parameters
# - Depth Transformer (DepFormer) parameters
# - Vocabulary sizes and audio settings
# - Delay structure for the delayed streams architecture


def config1b_202412() -> LmConfig:
    """
    Create configuration for the 1B parameter Moshi model (December 2024).
    
    This is a smaller, faster variant of Moshi suitable for:
    - Real-time inference on consumer hardware
    - Lower memory requirements
    - Faster generation with acceptable quality
    
    Architecture:
    - Temporal Transformer: 16 layers, d_model=2048, 16 heads (~1B params)
    - DepFormer: 6 layers, d_model=1024, 16 heads
    - 8 codebooks per speaker (16 total for full-duplex)
    - Delays: [0, 2, 2, 2, 2, 2, 2, 2] for each speaker
    
    Returns:
        LmConfig for the 1B model
    """
    transformer = TransformerConfig(
        d_model=2048,
        num_heads=16,
        num_layers=16,
        dim_feedforward=2048 * 4,  # dim * hidden_scale
        causal=True,
        norm_first=True,
        bias_ff=False,
        bias_attn=False,
        layer_scale=None,
        context=3000,
        max_period=100000,
        use_conv_block=False,
        use_conv_bias=True,
        cross_attention=False,
        gating=True,
        norm="rms_norm",
        positional_embedding="rope",
        conv_layout=False,
        conv_kernel_size=3,
        kv_repeat=1,
        max_seq_len=4096,
    )
    depformer = DepFormerConfig(
        transformer=TransformerConfig(
            d_model=1024,
            num_heads=16,
            num_layers=6,
            dim_feedforward=1024 * 4,  # dim * hidden_scale
            causal=True,
            norm_first=True,
            bias_ff=False,
            bias_attn=False,
            layer_scale=None,
            context=8,
            max_period=10000,
            use_conv_block=False,
            use_conv_bias=True,
            cross_attention=False,
            gating=True,
            norm="rms_norm",
            positional_embedding="none",
            conv_layout=False,
            conv_kernel_size=3,
            kv_repeat=1,
            max_seq_len=4096,
        ),
        num_slices=8,
    )
    return LmConfig(
        transformer=transformer,
        depformer=depformer,
        audio_vocab_size=2049,
        text_in_vocab_size=48001,
        text_out_vocab_size=48000,
        audio_codebooks=16,
        audio_delays=([0] + [2] * 7) * 2,
        conditioners={},
    )


def config1b_202412_16rvq() -> LmConfig:
    """
    Create configuration for the 1B model with 16 RVQ codebooks per speaker.
    
    This variant uses more codebooks for higher audio quality:
    - 16 codebooks per speaker (32 total) vs 8 in standard config
    - Higher bitrate audio representation
    - Better reconstruction quality at the cost of more computation
    
    Architecture:
    - Temporal Transformer: Same as config1b_202412
    - DepFormer: 16 slices (one per codebook), context=16
    - 16 codebooks per speaker (32 total for full-duplex)
    - Delays: [0, 2, 2, ..., 2] (15 delays of 2) for each speaker
    
    Returns:
        LmConfig for the 1B model with 16 RVQ levels
    """
    transformer = TransformerConfig(
        d_model=2048,
        num_heads=16,
        num_layers=16,
        dim_feedforward=2048 * 4,  # dim * hidden_scale
        causal=True,
        norm_first=True,
        bias_ff=False,
        bias_attn=False,
        layer_scale=None,
        context=3000,
        max_period=100000,
        use_conv_block=False,
        use_conv_bias=True,
        cross_attention=False,
        gating=True,
        norm="rms_norm",
        positional_embedding="rope",
        conv_layout=False,
        conv_kernel_size=3,
        kv_repeat=1,
        max_seq_len=4096,
    )
    depformer = DepFormerConfig(
        transformer=TransformerConfig(
            d_model=1024,
            num_heads=16,
            num_layers=6,
            dim_feedforward=1024 * 4,  # dim * hidden_scale
            causal=True,
            norm_first=True,
            bias_ff=False,
            bias_attn=False,
            layer_scale=None,
            context=16,
            max_period=10000,
            use_conv_block=False,
            use_conv_bias=True,
            cross_attention=False,
            gating=True,
            norm="rms_norm",
            positional_embedding="none",
            conv_layout=False,
            conv_kernel_size=3,
            kv_repeat=1,
            max_seq_len=4096,
        ),
        num_slices=16,
    )
    return LmConfig(
        transformer=transformer,
        depformer=depformer,
        audio_vocab_size=2049,
        text_in_vocab_size=48001,
        text_out_vocab_size=48000,
        audio_codebooks=32,
        audio_delays=([0] + [2] * 15) * 2,
        conditioners={},
    )


def config_v0_1() -> LmConfig:
    """
    Create configuration for the original Moshi v0.1 (7B parameters).
    
    This is the full-size Moshi model as described in the paper:
    - Highest quality generation
    - Requires significant GPU memory (~24GB+)
    - Best for research and high-quality applications
    
    Architecture:
    - Temporal Transformer: 32 layers, d_model=4096, 32 heads (~7B params)
    - DepFormer: 6 layers, d_model=1024, 16 heads
    - 8 codebooks per speaker (16 total for full-duplex)
    - Delays: [0, 1, 1, 1, 1, 1, 1, 1] for each speaker
    - Text vocabulary: 32000 (SentencePiece unigram)
    
    Note: The delay of 1 (vs 2 in newer configs) means tighter
    temporal coupling between semantic and acoustic tokens.
    
    Returns:
        LmConfig for the original 7B Moshi model
    """
    transformer = TransformerConfig(
        d_model=4096,
        num_heads=32,
        num_layers=32,
        dim_feedforward=4096 * 4,  # dim * hidden_scale
        causal=True,
        norm_first=True,
        bias_ff=False,
        bias_attn=False,
        layer_scale=None,
        context=3000,
        max_period=10000,
        use_conv_block=False,
        use_conv_bias=True,
        cross_attention=False,
        gating=True,
        norm="rms_norm",
        positional_embedding="rope",
        conv_layout=False,
        conv_kernel_size=3,
        kv_repeat=1,
        max_seq_len=4096,
    )
    depformer = DepFormerConfig(
        transformer=TransformerConfig(
            d_model=1024,
            num_heads=16,
            num_layers=6,
            dim_feedforward=1024 * 4,  # dim * hidden_scale
            causal=True,
            norm_first=True,
            bias_ff=False,
            bias_attn=False,
            layer_scale=None,
            context=8,
            max_period=10000,
            use_conv_block=False,
            use_conv_bias=True,
            cross_attention=False,
            gating=True,
            norm="rms_norm",
            positional_embedding="none",
            conv_layout=False,
            conv_kernel_size=3,
            kv_repeat=1,
            max_seq_len=4096,
        ),
        num_slices=8,
    )
    return LmConfig(
        transformer=transformer,
        depformer=depformer,
        audio_vocab_size=2049,
        text_in_vocab_size=32001,
        text_out_vocab_size=32000,
        audio_codebooks=16,
        audio_delays=([0] + [1] * 7) * 2,
        conditioners={},
    )


def config_helium_1_preview_2b() -> LmConfig:
    """
    Create configuration for the Helium text-only model (2B parameters).
    
    Helium is the pure TEXT backbone that Moshi is built upon.
    This configuration is for text-only inference without audio.
    
    Key differences from full Moshi:
    - NO DepFormer (num_slices=0)
    - NO audio codebooks
    - NO audio delays
    - Pure autoregressive text generation
    
    Use cases:
    - Text-only language modeling
    - Initializing Moshi's Temporal Transformer
    - Benchmarking text capabilities
    
    Architecture:
    - Transformer: 24 layers, d_model=2560, 20 heads (~2B params)
    - Context: 4096 tokens
    - Text vocabulary: 48000
    
    Returns:
        LmConfig for the text-only Helium model
    """
    transformer = TransformerConfig(
        d_model=2560,
        num_heads=20,
        num_layers=24,
        dim_feedforward=2560 * 4,  # dim * hidden_scale
        causal=True,
        norm_first=True,
        bias_ff=False,
        bias_attn=False,
        layer_scale=None,
        context=4096,
        max_period=100000,
        use_conv_block=False,
        use_conv_bias=True,
        cross_attention=False,
        gating=True,
        norm="rms_norm",
        positional_embedding="rope",
        conv_layout=False,
        conv_kernel_size=3,
        kv_repeat=1,
        max_seq_len=4096,
    )
    depformer = DepFormerConfig(
        transformer=transformer,
        num_slices=0,
    )
    return LmConfig(
        transformer=transformer,
        depformer=depformer,
        audio_vocab_size=2049,
        text_in_vocab_size=48000,
        text_out_vocab_size=48000,
        audio_codebooks=0,
        audio_delays=[],
        conditioners={},
    )
