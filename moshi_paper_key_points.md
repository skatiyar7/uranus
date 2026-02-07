# Moshi: A Speech-Text Foundation Model for Real-Time Full-Duplex Dialogue

---

## 1. Overview

- **Project name:** Moshi
- **Type:** Speech-text foundation model
- **Primary capability:** Real-time, full-duplex, speech-to-speech dialogue
- **Latency target:**
  - Theoretical: ~160 ms
  - Practical: ~200 ms
- **Core novelty:** Joint modeling of text, semantic audio, and acoustic audio tokens across *multiple concurrent speaker streams* without turn segmentation

### Design Goals

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

## 2. Helium: Text Backbone

Helium is a **pure text autoregressive language model** whose role is to provide:
- linguistic structure (syntax, semantics),
- world knowledge,
- long-range reasoning and planning.

Helium **never predicts speech or audio tokens**.

### 2.1 Tokenization and Vocabulary

- **Tokenizer:** SentencePiece **unigram**
- **Vocabulary size:** **32,000**
- **Language focus:** English
- **Special handling:**
  - All numbers split into individual digits
  - Byte fallback for full information preservation
- **Training data:** ~2.1T tokens (English-centric)

**Important:** The vocabulary size **never changes**, including during Moshi training.

### 2.2 Architecture

Helium is a decoder-only Transformer:

| Component | Value |
|-----------|-------|
| Parameters | ~7B |
| Layers | 32 |
| Model dimension | 4096 |
| Attention heads | 32 |
| Context length | 4096 tokens |
| Normalization | RMSNorm (pre-norm everywhere) |
| FFN | GLU with SiLU activation |
| Positional encoding | RoPE |
| Attention kernel | FlashAttention |

### 2.3 Forward Pass (Text-Only Training)

Given text tokens:
```
w₀, w₁, ..., wₜ₋₁
```

1. **Embedding**
   ```
   x_t = E_text(w_t) ∈ ℝ⁴⁰⁹⁶
   ```

2. **Transformer stack**
   - 32 layers
   - Causal self-attention
   - RoPE applied in attention

3. **Hidden state**
   ```
   h_t ∈ ℝ⁴⁰⁹⁶
   ```

4. **Text logits (only during text training)**
   ```
   logits_t = W_vocab · h_t → ℝ³²⁰⁰⁰
   ```

### 2.4 What Helium Is Not

- ❌ No audio tokens
- ❌ No semantic tokens
- ❌ No acoustic tokens
- ❌ No multimodal vocabulary
- ❌ No speech generation

### 2.5 Purpose in Moshi

- Provides linguistic competence, world knowledge, and reasoning
- Initializes Moshi's Temporal Transformer
- Contributes **hidden states only** — text output head, logits, and softmax probabilities are **not used** when acting as Temporal Transformer

---

## 3. Mimi: Neural Audio Codec

Mimi is a causal neural audio codec that converts speech to discrete tokens and back.

### 3.1 Specifications

| Property | Value |
|----------|-------|
| Type | Causal neural audio codec with discrete tokens |
| Input | 24 kHz mono waveform |
| Output | Discrete token streams at 12.5 Hz |
| Quantization | Residual Vector Quantization (RVQ) |
| Total codebooks | 8 |
| Codebook size | 2048 |
| Bitrate | ~1.1 kbps |

### 3.2 Key Innovation: Split RVQ

- **1 semantic codebook** — distilled from WavLM, captures linguistic content
- **7 acoustic codebooks** — capture voice quality, prosody, acoustic details

### 3.3 Training Signals

- Adversarial loss (primary)
- Feature matching loss
- Semantic distillation loss (cosine similarity)

### 3.4 Properties

- Fully causal
- Streaming-compatible encoding and decoding

### 3.5 Purpose in Moshi

- Converts speech ↔ tokens
- Embeds linguistic (semantic) and acoustic detail in a single tokenizer

---

## 4. Moshi Architecture: RQ-Transformer

Moshi reuses Helium as a **Temporal Transformer** and augments it with:
- multimodal embedding sums,
- a **Depth Transformer** for intra-timestep modeling,
- a neural audio tokenizer (semantic + acoustic),
- multi-stream (user/system) modeling,
- **Inner Monologue** (text-guided speech generation).

Helium is **not replaced** — it is **wrapped in time and hierarchy**.

### 4.1 Two-Level Autoregressive Hierarchy

Moshi uses a **two-level autoregressive hierarchy**:

#### Temporal Transformer (Helium)
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

### 4.2 Acoustic Delay

- Semantic tokens are generated *ahead* of acoustic tokens
- Typical delay: 1–2 frames
- Effect:
  - Temporal Transformer captures semantic-acoustic dependencies
  - Improves stability and intelligibility
  - Enables streaming generation

---

## 5. Input Representation in Moshi

### 5.1 Parallel Token Streams per Timestep

At timestep `t`, Moshi maintains multiple parallel token streams:
- Text token (system, optional – Inner Monologue)
- Semantic audio token (RVQ level 1)
- Acoustic audio tokens (RVQ levels 2…Q, delayed)
- Corresponding user-side tokens

Formally:
```
V_t = { V_{t,1}, V_{t,2}, ..., V_{t,K} }
```

### 5.2 Embedding Tables

Each token type has its **own embedding table**:
```
E^(k): token_k → ℝ⁴⁰⁹⁶
```

No embeddings are shared across token types.

### 5.3 Summed Timestep Embedding

For timestep `t`:
```
x_t = Σ_k E^(k)(V_{t,k}) ∈ ℝ⁴⁰⁹⁶
```

- This summation happens **before Helium**
- Analogous to token + positional + segment embeddings in text models
- Helium sees **only the sum**, not individual token identities

---

## 6. Temporal Transformer: Helium Inside Moshi

### 6.1 Input to Helium

At timestep `s`, Helium receives:
```
X_<s = [x₀, x₁, ..., x_{s-1}] ∈ ℝ^{s × 4096}
```

- Causal mask
- RoPE positional encoding
- No awareness of modality types

### 6.2 Helium Computation

Helium performs a **standard Transformer forward pass** over `X_<s`.

Final hidden states:
```
H^(32) ∈ ℝ^{s × 4096}
```

### 6.3 The Shared Output: Context Vector

Moshi extracts:
```
z_s = H^(32)[s-1] ∈ ℝ⁴⁰⁹⁶
```

This vector is:
- last layer,
- last timestep,
- before any output projection.

**This is the only information passed from Helium to the rest of Moshi.**

### 6.4 What Is Not Used from Helium

When Helium acts as the Temporal Transformer:
- ❌ Text output head
- ❌ Text logits
- ❌ Softmax probabilities

Helium contributes **hidden states only**, never logits.

---

## 7. Depth Transformer: Intra-Timestep Modeling

### 7.1 Purpose

The Depth Transformer models **structure within a single timestep**, avoiding long flattened sequences.

It predicts:
- text tokens (Inner Monologue),
- semantic audio tokens,
- acoustic RVQ tokens.

### 7.2 Architecture

| Component | Value |
|-----------|-------|
| Layers | 6 |
| Model dimension | 1024 |
| Attention heads | 16 |

### 7.3 Inputs

For timestep `s` and subtoken index `k`:
```
input_{s,k} = z_s + E^(k)(V_{s,k-1})
```

- `z_s`: global temporal context from Helium
- `E^(k)`: local token-type embedding

### 7.4 Outputs

For each token type `k`, the Depth Transformer produces logits:
```
ℓ_{s,k} ∈ ℝ^{|V_k|}
```

Examples:
- Text tokens → 32k-way softmax
- Semantic audio → 2048-way softmax
- Acoustic RVQ levels → 2048-way softmax per level

**All token prediction happens here, not in Helium.**

---

## 8. Multi-Stream Modeling (Full-Duplex)

Moshi models **two concurrent audio streams**:
1. **System (Moshi)**
2. **User**

Each stream has:
- Semantic token sequence
- Acoustic token sequence

Streams are modeled *jointly* in a single autoregressive process.
- No explicit turn boundaries
- Silence is represented as natural audio, not special tokens

### 8.1 Joint Token Layout (Per Time Step)

For each timestep `s`, Moshi models:

1. Text tokens (Inner Monologue, Moshi only)
2. Semantic tokens (Moshi)
3. Delayed acoustic tokens (Moshi)
4. Semantic tokens (User)
5. Delayed acoustic tokens (User)

Total streams:
- `K = 2Q + 1` (with Q = 8 → K = 17)

### 8.2 Outcome

- Overlapping speech
- Interruptions
- Backchannels
- Continuous listening while speaking

---

## 9. Inner Monologue (Text-Audio Alignment)

### 9.1 Definition

Inner Monologue is a training & inference scheme where **Moshi predicts time-aligned text tokens as a prefix to its own audio tokens**.

### 9.2 Hierarchical Decomposition

Inner Monologue decomposes speech generation hierarchically:
```
Text → Semantic Audio → Acoustic Audio
```

Text tokens are produced **incrementally per frame**, not as a full sequence first.

### 9.3 Mechanics

- Text tokens aligned to 12.5 Hz using Whisper word timestamps
- Special tokens:
  - `PAD` – padding between words
  - `EPAD` – explicit end-of-padding marker
- Text stream applies **only to Moshi's speech**, not the user

### 9.4 Benefits

1. Stronger linguistic coherence
2. Improved factuality
3. Longer, more consistent speech generations
4. Streaming-compatible speech generation
5. Enables *derived tasks* via delay manipulation:
   - **Streaming ASR:** audio → delayed text
   - **Streaming TTS:** text → delayed audio

---

## 10. Training Pipeline

### Stage 1 — Helium Pretraining
- Text-only LLM training
- 500k steps
- No embedding sums
- Standard LM training

### Stage 2 — Moshi Audio Pretraining
- Single audio stream
- Unsupervised ~7M hours
- Random text-audio delays
- Mixed text-only batches to prevent forgetting
- Helium initialized from pretrained checkpoint
- Helium **does see summed embeddings**
- Safeguards:
  - continued pure-text batches,
  - separate optimizer states,
  - reduced learning rate on text parameters
- Goal: Teach Helium to **accept multimodal context vectors without losing language ability**

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

## 11. What Is Shared vs Not Shared

### Shared
- Helium weights (initialization)
- Final hidden-state context vector `z_s`

### Not Shared
- ❌ Vocabularies
- ❌ Logits
- ❌ Softmax outputs
- ❌ Audio prediction responsibilities

---

## 12. Capabilities Summary

- Real-time speech-to-speech dialogue
- Full-duplex interaction
- Long-context reasoning (~5 minutes)
- Streaming ASR and TTS from the same model
- Paralinguistic understanding and generation
- Arbitrary voices, emotions, and acoustic conditions

---

## 13. Core Design Principle

**Helium remains a pure text reasoner. Moshi turns Helium's hidden states into a real-time, hierarchical, multimodal dialogue system.**

---

## 14. Scope Notes

- Appendix sections intentionally excluded
- Safety, extended evaluations, and deployment details omitted
- Document reflects *core system design and training only*

---
