# Moshi Manifest
*A Speech-Text Foundation Model for Real-Time Full-Duplex Dialogue*

---

## 1. Identity

- **Project name:** Moshi  
- **Type:** Speech-text foundation model  
- **Primary capability:** Real-time, full-duplex, speech-to-speech dialogue  
- **Latency target:**  
  - Theoretical: ~160 ms  
  - Practical: ~200 ms  
- **Core novelty:** Joint modeling of text, semantic audio, and acoustic audio tokens across *multiple concurrent speaker streams* without turn segmentation

---

## 2. Design Goals

1. **Real-time interaction**
   - Sub-second response latency
   - Streaming inference throughout the pipeline

2. **Speech-native reasoning**
   - No mandatory ASR → text → TTS cascade
   - Preserve paralinguistic information (emotion, prosody, non-speech sounds)

3. **Full-duplex dialogue**
   - Always listening, always able to speak
   - Supports overlap, interruptions, backchanneling

4. **LLM-level reasoning**
   - Inherit knowledge, factuality, and reasoning from a large text LLM

---

## 3. System Components

### 3.1 Helium (Text Backbone)

- **Type:** Autoregressive Transformer LLM
- **Parameters:** ~7B
- **Context length:** 4096 tokens
- **Tokenizer:** SentencePiece unigram (32k vocab, byte-fallback)
- **Architecture details:**
  - RMSNorm
  - RoPE positional embeddings
  - FlashAttention
  - Gated Linear Units (SiLU)
- **Training data:** ~2.1T tokens (English-centric)
- **Purpose in Moshi:**
  - Provides linguistic competence, world knowledge, and reasoning
  - Initializes Moshi’s Temporal Transformer

---

### 3.2 Mimi (Neural Audio Codec)

- **Type:** Causal neural audio codec with discrete tokens
- **Input:** 24 kHz mono waveform
- **Output:** Discrete token streams at 12.5 Hz
- **Compression:**
  - Residual Vector Quantization (RVQ)
  - 8 total codebooks
  - Codebook size: 2048
  - Bitrate: ~1.1 kbps
- **Key innovation: Split RVQ**
  - 1 semantic codebook (distilled from WavLM)
  - 7 acoustic codebooks
- **Training signals:**
  - Adversarial loss (primary)
  - Feature matching loss
  - Semantic distillation loss (cosine similarity)
- **Properties:**
  - Fully causal
  - Streaming-compatible encoding and decoding
- **Purpose in Moshi:**
  - Converts speech ↔ tokens
  - Embeds linguistic (semantic) and acoustic detail in a single tokenizer

---

## 4. Generative Architecture

### 4.1 RQ-Transformer (Hierarchical Generation)

Moshi uses a **two-level autoregressive hierarchy**:

#### Temporal Transformer
- Large model (initialized from Helium)
- Operates over *time steps*
- Produces a temporal context embedding per frame

#### Depth Transformer
- Smaller Transformer
- Operates over *token streams within a frame*
- Predicts:
  - Text tokens (optional, via Inner Monologue)
  - Semantic audio tokens
  - Acoustic audio tokens

**Benefit:**
- Avoids flattening all audio tokens into one long sequence
- Enables long-context + real-time generation

---

### 4.2 Acoustic Delay

- Semantic tokens are generated *ahead* of acoustic tokens
- Typical delay: 1–2 frames
- Effect:
  - Temporal Transformer captures semantic-acoustic dependencies
  - Improves stability and intelligibility
  - Enables streaming generation

---

## 5. Multi-Stream Modeling (Full-Duplex)

- Moshi models **two concurrent audio streams**:
  1. **System (Moshi)**
  2. **User**
- Each stream has:
  - Semantic token sequence
  - Acoustic token sequence
- Streams are modeled *jointly* in a single autoregressive process
- No explicit turn boundaries
- Silence is represented as natural audio, not special tokens

**Outcome:**
- Overlapping speech
- Interruptions
- Backchannels
- Continuous listening while speaking

---

## 6. Inner Monologue (Text-Audio Alignment)

### Definition
Inner Monologue is a training & inference scheme where **Moshi predicts time-aligned text tokens as a prefix to its own audio tokens**.

### Mechanics
- Text tokens aligned to 12.5 Hz using Whisper word timestamps
- Special tokens:
  - `PAD` – padding between words
  - `EPAD` – explicit end-of-padding marker
- Text stream applies **only to Moshi’s speech**, not the user

### Benefits
1. Stronger linguistic coherence
2. Improved factuality
3. Longer, more consistent speech generations
4. Enables *derived tasks* via delay manipulation:
   - **Streaming ASR:** audio → delayed text
   - **Streaming TTS:** text → delayed audio

---

## 7. Joint Token Layout (Per Time Step)

For each timestep `s`, Moshi models:

1. Text tokens (Inner Monologue, Moshi only)
2. Semantic tokens (Moshi)
3. Delayed acoustic tokens (Moshi)
4. Semantic tokens (User)
5. Delayed acoustic tokens (User)

Total streams:
- `K = 2Q + 1` (with Q = 8 → K = 17)

---

## 8. Training Pipeline

### Stage 1 — Helium Pretraining
- Text-only LLM training
- 500k steps

### Stage 2 — Moshi Audio Pretraining
- Single audio stream
- Unsupervised ~7M hours
- Random text-audio delays
- Mixed text-only batches to prevent forgetting

### Stage 3 — Multi-Stream Post-Training
- Simulated speaker separation via diarization
- Two audio streams
- No overlap yet

### Stage 4 — Full-Duplex Finetuning
- Fisher dataset (true two-channel conversations)
- Natural overlap and interruptions

### Stage 5 — Instruction Finetuning
- Synthetic speech conversations
- Controlled Moshi voice
- Robustness augmentations (noise, echo, gain, reverb)

---

## 9. Capabilities Summary

- Real-time speech-to-speech dialogue
- Full-duplex interaction
- Long-context reasoning (~5 minutes)
- Streaming ASR and TTS from the same model
- Paralinguistic understanding and generation
- Arbitrary voices, emotions, and acoustic conditions

---

## 10. Scope Notes

- Appendix sections intentionally excluded
- Safety, extended evaluations, and deployment details omitted
- Manifest reflects *core system design and training only*

---
