# Moshi MLX - Neural Audio Generation on Apple Silicon

This package provides the MLX implementation for running inference with Kyutai's Moshi audio generation models on Apple Silicon (M1/M2/M3/M4) hardware.

## Overview

Moshi is a real-time conversational AI model that can engage in spoken dialogue. It jointly models text and audio using a "delayed streams" architecture, enabling natural voice conversations with low latency.

## Package Structure

```
moshi_mlx/
├── __init__.py          # Package entry point
├── local.py             # Local audio inference (microphone/speakers)
├── local_web.py         # Web-based inference server (WebSocket)
├── client_utils.py      # Terminal output utilities
├── run_helium.py        # Helium text-only model runner
├── run_inference.py     # Offline audio file processing
├── run_tts.py           # Text-to-speech synthesis
│
├── models/              # High-level model implementations
│   ├── __init__.py
│   ├── lm.py            # Language model (Lm, LmConfig)
│   ├── generate.py      # Generation wrapper (LmGen)
│   ├── mimi.py          # Audio codec (Mimi)
│   └── tts.py           # TTS model and state machine
│
├── modules/             # Neural network building blocks
│   ├── __init__.py
│   ├── transformer.py   # Transformer layers and attention
│   ├── kv_cache.py      # Key-value caching for generation
│   ├── conv.py          # 1D convolution layers (streaming)
│   ├── quantization.py  # Vector quantization (RVQ)
│   ├── seanet.py        # SEANet encoder/decoder
│   └── conditioner.py   # Conditioning mechanisms
│
└── utils/               # Utility functions
    ├── __init__.py
    ├── sampling.py      # Token sampling strategies
    └── loaders.py       # File loading utilities
```

## Module Relationships

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Entry Points                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  local.py          local_web.py       run_inference.py      run_tts.py      │
│  (Microphone)      (WebSocket)        (File I/O)            (TTS)           │
└────────┬─────────────────┬────────────────┬─────────────────────┬───────────┘
         │                 │                │                     │
         └────────────┬────┴────────────────┴─────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Models Layer                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐              │
│   │    Mimi      │      │     Lm       │      │    LmGen     │              │
│   │ (Audio Codec)│      │ (Language    │      │ (Generation  │              │
│   │              │◄────►│   Model)     │◄────►│   Wrapper)   │              │
│   └──────────────┘      └──────────────┘      └──────────────┘              │
│         │                      │                     │                       │
│         │                      │                     │                       │
│   Encode/Decode          Text + Audio           Delayed Streams              │
│   Audio ↔ Tokens         Generation             Management                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Modules Layer                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│   │Transformer │  │  SEANet    │  │    RVQ     │  │ Conditioner│            │
│   │            │  │            │  │            │  │            │            │
│   │ • Attention│  │ • Encoder  │  │ • Codebook │  │ • LUT      │            │
│   │ • FFN      │  │ • Decoder  │  │ • Quantize │  │ • Tensor   │            │
│   │ • KV Cache │  │ • Residual │  │ • Decode   │  │            │            │
│   └────────────┘  └────────────┘  └────────────┘  └────────────┘            │
│                                                                              │
│   ┌────────────┐  ┌────────────┐                                            │
│   │   Conv1D   │  │  KV Cache  │                                            │
│   │            │  │            │                                            │
│   │ • Standard │  │ • Standard │                                            │
│   │ • Streaming│  │ • Rotating │                                            │
│   └────────────┘  └────────────┘                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Utils Layer                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│   ┌────────────────────┐      ┌────────────────────┐                        │
│   │      Sampler       │      │      Loaders       │                        │
│   │                    │      │                    │                        │
│   │ • Top-K            │      │ • HuggingFace Hub  │                        │
│   │ • Top-P (Nucleus)  │      │ • Local files      │                        │
│   │ • Min-P            │      │ • hf:// URLs       │                        │
│   │ • Temperature      │      │                    │                        │
│   └────────────────────┘      └────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Request Flow Diagram (Streaming Session)

The following diagram shows the flow of audio data during a real-time streaming session using `local_web.py`:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STREAMING SESSION FLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

    Browser/Client                Web Server Process              Model Server Process
    ══════════════                ══════════════════              ════════════════════
          │                              │                               │
          │  1. WebSocket Connect        │                               │
          │─────────────────────────────►│                               │
          │                              │                               │
          │  2. Handshake (0x00)         │                               │
          │◄─────────────────────────────│                               │
          │                              │                               │
          │                              │                               │
    ┌─────┴─────┐                  ┌─────┴─────┐                   ┌─────┴─────┐
    │  AUDIO    │                  │   AUDIO   │                   │   MODEL   │
    │  INPUT    │                  │  PIPELINE │                   │ INFERENCE │
    │  LOOP     │                  │           │                   │   LOOP    │
    └─────┬─────┘                  └─────┬─────┘                   └─────┬─────┘
          │                              │                               │
          │                              │                               │
          ▼                              ▼                               ▼
┌─────────────────┐            ┌─────────────────┐             ┌─────────────────┐
│ 3. Capture Mic  │            │ 4. Opus Decode  │             │                 │
│    Audio        │───────────►│    to PCM       │             │                 │
│    (Opus)       │  0x01+data │    (24kHz)      │             │                 │
└─────────────────┘            └────────┬────────┘             │                 │
                                        │                      │                 │
                                        ▼                      │                 │
                               ┌─────────────────┐             │                 │
                               │ 5. Mimi Encode  │             │                 │
                               │    PCM → Tokens │             │                 │
                               │    (1920 samples│             │                 │
                               │     = 80ms)     │             │                 │
                               └────────┬────────┘             │                 │
                                        │                      │                 │
                                        │  Audio Tokens        │                 │
                                        │  via Queue           │                 │
                                        │─────────────────────►│ 6. LmGen.step() │
                                        │                      │    Generate     │
                                        │                      │    text + audio │
                                        │                      │    tokens       │
                                        │                      └────────┬────────┘
                                        │                               │
                                        │  Generated Tokens             │
                                        │  (text + audio)               │
                                        │◄──────────────────────────────│
                                        │                               │
                               ┌────────┴────────┐                      │
                               │ 7. Mimi Decode  │                      │
                               │    Tokens → PCM │                      │
                               └────────┬────────┘                      │
                                        │                               │
                               ┌────────┴────────┐                      │
                               │ 8. Opus Encode  │                      │
                               │    PCM → Opus   │                      │
                               └────────┬────────┘                      │
                                        │                               │
          ┌─────────────────────────────┘                               │
          │  0x01+audio / 0x02+text                                     │
          ▼                                                             │
┌─────────────────┐                                                     │
│ 9. Play Audio   │                                                     │
│    Display Text │                                                     │
└─────────────────┘                                                     │
          │                                                             │
          │         (Loop continues for duration of session)            │
          └─────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                         DETAILED TOKEN FLOW                                  │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │         User's Speech               │
                    │      (Microphone Input)             │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │         Mimi Encoder                │
                    │   PCM (24kHz) → Audio Tokens        │
                    │   [8-32 codebooks × T frames]       │
                    └──────────────┬──────────────────────┘
                                   │
                                   │ other_audio_tokens
                                   │ (user's speech tokens)
                                   ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │                        LmGen.step()                               │
    │  ┌────────────────────────────────────────────────────────────┐  │
    │  │                    Main Transformer                         │  │
    │  │                                                             │  │
    │  │   Input: text_emb + Σ(audio_emb[i] for i in codebooks)     │  │
    │  │                                                             │  │
    │  │   ┌─────────────┐                                           │  │
    │  │   │ Text Token  │◄── Previous text token (or start token)  │  │
    │  │   │ Embedding   │                                           │  │
    │  │   └──────┬──────┘                                           │  │
    │  │          │                                                  │  │
    │  │          ▼                                                  │  │
    │  │   ┌─────────────┐                                           │  │
    │  │   │   + Audio   │◄── Delayed audio tokens from buffer      │  │
    │  │   │  Embeddings │    (accounts for audio_delays)           │  │
    │  │   └──────┬──────┘                                           │  │
    │  │          │                                                  │  │
    │  │          ▼                                                  │  │
    │  │   ┌─────────────┐                                           │  │
    │  │   │ Transformer │    32 layers, 4096 dim                   │  │
    │  │   │   Layers    │    (with KV caching)                     │  │
    │  │   └──────┬──────┘                                           │  │
    │  │          │                                                  │  │
    │  │          ▼                                                  │  │
    │  │   ┌─────────────┐                                           │  │
    │  │   │ Text Linear │───► text_token (sampled)                 │  │
    │  │   └─────────────┘                                           │  │
    │  └─────────────────────────────┬──────────────────────────────┘  │
    │                                │                                  │
    │                                │ transformer_out                  │
    │                                ▼                                  │
    │  ┌────────────────────────────────────────────────────────────┐  │
    │  │                      DepFormer                              │  │
    │  │                                                             │  │
    │  │   For each codebook slice (0 to num_slices-1):             │  │
    │  │                                                             │  │
    │  │   ┌─────────────┐                                           │  │
    │  │   │ Slice Input │◄── text_token (slice 0) or               │  │
    │  │   │  Embedding  │    prev_audio_token (slice 1+)           │  │
    │  │   └──────┬──────┘                                           │  │
    │  │          │                                                  │  │
    │  │          ▼                                                  │  │
    │  │   ┌─────────────┐                                           │  │
    │  │   │  + Linear   │◄── transformer_out projection            │  │
    │  │   │   (from     │                                           │  │
    │  │   │   main TF)  │                                           │  │
    │  │   └──────┬──────┘                                           │  │
    │  │          │                                                  │  │
    │  │          ▼                                                  │  │
    │  │   ┌─────────────┐                                           │  │
    │  │   │DepFormer TF │    6 layers, 1024 dim                    │  │
    │  │   └──────┬──────┘                                           │  │
    │  │          │                                                  │  │
    │  │          ▼                                                  │  │
    │  │   ┌─────────────┐                                           │  │
    │  │   │Audio Linear │───► audio_token[slice] (sampled)         │  │
    │  │   └─────────────┘                                           │  │
    │  │                                                             │  │
    │  └────────────────────────────────────────────────────────────┘  │
    │                                                                   │
    │   Output: text_token, audio_tokens[0:num_slices]                 │
    └───────────────────────────────────┬──────────────────────────────┘
                                        │
                                        │ audio_tokens (model's speech)
                                        ▼
                    ┌─────────────────────────────────────┐
                    │         Mimi Decoder                │
                    │   Audio Tokens → PCM (24kHz)        │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │        Model's Speech               │
                    │       (Speaker Output)              │
                    └─────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                      DELAYED STREAMS VISUALIZATION                           │
└─────────────────────────────────────────────────────────────────────────────┘

The model uses "delayed streams" where audio codebooks have different delays
relative to the text stream. This allows proper temporal alignment:

Time Step:        0    1    2    3    4    5    6    7    8    ...
                  │    │    │    │    │    │    │    │    │
Text Stream:      T0   T1   T2   T3   T4   T5   T6   T7   T8   ...
                  │    │    │    │    │    │    │    │    │
Audio CB 0:       ·    A0   A1   A2   A3   A4   A5   A6   A7   ... (delay=1)
Audio CB 1:       ·    A0   A1   A2   A3   A4   A5   A6   A7   ... (delay=1)
Audio CB 2:       ·    A0   A1   A2   A3   A4   A5   A6   A7   ... (delay=1)
...
Audio CB 7:       ·    A0   A1   A2   A3   A4   A5   A6   A7   ... (delay=1)
Audio CB 8:       ·    ·    A0   A1   A2   A3   A4   A5   A6   ... (delay=2)
...

Legend:
  T = Text token
  A = Audio token
  · = Padding token (not yet generated)

At each step, the model:
1. Receives user's audio tokens (other_audio_tokens)
2. Retrieves delayed audio tokens from the generation buffer
3. Generates new text token
4. Generates new audio tokens via DepFormer
5. Stores generated tokens at their delayed positions
```

## Key Components

### Models

| Component | Description |
|-----------|-------------|
| **Lm** | Main language model combining transformer + depformer for joint text/audio generation |
| **LmGen** | Generation wrapper handling delayed streams, sampling, and sequence management |
| **Mimi** | Neural audio codec (encoder/decoder) converting audio ↔ discrete tokens |
| **TTSModel** | Text-to-speech wrapper with state machine for DSM-based synthesis |

### Modules

| Component | Description |
|-----------|-------------|
| **Transformer** | Multi-head attention with RoPE, KV caching, and optional cross-attention |
| **SEANet** | Convolutional encoder/decoder for audio compression |
| **RVQ** | Residual Vector Quantization for discretizing audio representations |
| **KVCache** | Key-value caching for efficient autoregressive generation |
| **Conditioner** | Conditioning mechanisms (LUT for quality, tensor for speaker) |

### Utils

| Component | Description |
|-----------|-------------|
| **Sampler** | Token sampling strategies (top-k, top-p, min-p, temperature) |
| **Loaders** | File loading with HuggingFace Hub integration |

## Usage Examples

### Web-based Inference
```bash
python -m moshi_mlx.local_web --hf-repo kyutai/moshiko-mlx-q8
```

### Local Audio Inference
```bash
python -m moshi_mlx.local --hf-repo kyutai/moshiko-mlx-q8
```

### Offline File Processing
```bash
python -m moshi_mlx.run_inference input.wav output.wav
```

### Text Generation (Helium)
```bash
python -m moshi_mlx.run_helium --prompt "Hello, world"
```

## Configuration

Models can be configured via:
- `config_v0_1()` - Original Moshi 7B configuration
- `config1b_202412()` - 1B parameter model
- `config_helium_1_preview_2b()` - Helium text-only 2B model

Quantization options:
- `--quantized 8` - 8-bit quantization (recommended)
- `--quantized 4` - 4-bit quantization (faster, lower quality)

## Audio Specifications

| Parameter | Value |
|-----------|-------|
| Sample Rate | 24,000 Hz |
| Frame Size | 1,920 samples (80ms) |
| Frame Rate | 12.5 Hz |
| Channels | 1 (mono) |
| Codebooks | 8-32 (configurable) |
| Codebook Size | 2,048 entries |

## License

Copyright (c) Kyutai, all rights reserved.
See the LICENSE file in the root directory for details.
