# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Text-to-Speech (TTS) Module using Delayed Streams Modeling
==========================================================

This module implements a TTS system built on top of Moshi's Delayed Streams
Modeling (DSM) architecture. Unlike traditional TTS that generates audio from
a fixed text sequence, DSM-based TTS co-generates the time-aligned text
representation along with the audio.

=============================================================================
THE TTS CHALLENGE WITH DELAYED STREAMS
=============================================================================

In Moshi's architecture, text and audio are generated together at each timestep.
For speech-to-text (STT), this is straightforward: we feed audio tokens and
sample text tokens.

For TTS, the challenge is:
- We start with UNPACED text (just words, no timing)
- We need to generate PACED text (time-aligned with audio)
- The model must decide WHEN to start each word

The solution is a STATE MACHINE that:
1. Monitors the model's text predictions
2. When the model predicts "new_word", feeds the next word's tokens
3. Controls pacing through padding constraints

=============================================================================
STATE MACHINE OVERVIEW
=============================================================================

The TTS state machine uses special tokens:
- new_word (0): Model requests the next word
- main (1): Start of main speaker's turn
- other (2): Start of other speaker's turn  
- pad (3): Padding (silence/continuation)

Generation flow:
1. Model predicts a token (new_word or pad)
2. State machine processes the prediction:
   - If new_word: Pop next word from queue, feed its tokens
   - If pad: Continue with padding or queued tokens
3. State machine returns the actual input for next step

Constraints:
- max_padding: Maximum consecutive pads (prevents too slow speech)
- forced_padding: Minimum pads after a word (prevents too fast speech)
- initial_padding: Pads at the start (prevents cut-off)

=============================================================================
MULTI-SPEAKER SUPPORT
=============================================================================

The TTS model supports multiple speakers through:
1. Speaker tokens (main/other) at turn boundaries
2. Voice conditioning via cross-attention
3. Pre-computed speaker embeddings

Voice embeddings are loaded from .safetensors files and provide:
- Speaker identity (voice characteristics)
- Speaking style
- Prosody patterns

=============================================================================
CLASSIFIER-FREE GUIDANCE (CFG)
=============================================================================

Two CFG modes are supported:
1. Direct CFG: Compute conditional and unconditional logits at inference.
   NOT supported by CFG-distilled models.
2. CFG Distillation: Model was trained with CFG baked in.
   Pass cfg_coef to make_condition_attributes() instead of using it
   directly in generation.

For distilled models, pass cfg_coef to make_condition_attributes() instead
of using it directly in generation.

=============================================================================
KEY CLASSES
=============================================================================

- TokenIds: Special token values for the state machine
- Entry: One word to generate (tokens + metadata)
- State: Current state of the TTS generation
- StateMachine: Controls text feeding based on model predictions
- TTSResult: Output of TTS generation
- TTSModel: Main TTS interface wrapping LM + Mimi + tokenizer

=============================================================================
USAGE EXAMPLE
=============================================================================

    # Load model
    tts = TTSModel.from_checkpoint_info(...)
    
    # Prepare script (list of turns)
    script = ["Hello, how are you?", "I'm doing great!"]
    entries = tts.prepare_script(script)
    
    # Get voice conditioning
    voice_path = tts.get_voice_path("alice")
    attributes = tts.make_condition_attributes([voice_path])
    
    # Generate
    result = tts.generate([entries], [attributes])
    
    # Decode audio
    for frame in result.frames:
        audio = mimi.decode_step(frame)
"""

import re
import typing as tp
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import mlx.core as mx
import sphn
from sentencepiece import SentencePieceProcessor

from ..modules.conditioner import (
    ConditionAttributes,
    ConditionTensor,
    LutConditioner,
    TensorCondition,
    dropout_all_conditions,
)
from ..utils.loaders import hf_get
from ..utils.sampling import Sampler
from . import Lm, LmGen
from .mimi import Mimi

DEFAULT_DSM_TTS_REPO = "kyutai/tts-1.6b-en_fr"
DEFAULT_DSM_TTS_VOICE_REPO = "kyutai/tts-voices"


@dataclass
class TokenIds:
    """
    Special token IDs for the TTS state machine.
    
    These tokens control the flow of text feeding during TTS generation.
    The state machine monitors model predictions and uses these tokens
    to coordinate when to feed new words.
    
    ==========================================================================
    TOKEN SEMANTICS
    ==========================================================================
    
    CONTROL TOKENS (predicted by model, processed by state machine):
    - new_word (0): Model requests the next word to be fed
    - main (1): Indicates start of main speaker's turn
    - other (2): Indicates start of other speaker's turn
    - pad (3): Padding token (silence/continuation)
    
    SPECIAL VALUES (used internally):
    - zero (-1): Produces zero embedding (no input)
    - ungenerated (-2): Marks positions not yet generated
    
    ==========================================================================
    MULTIPLEXING (for second_stream_ahead)
    ==========================================================================
    
    When using lookahead (second_stream_ahead > 0), two tokens are
    multiplexed into one value:
    
        combined = (second_token + 1) * card + main_token
    
    The +1 allows encoding -1 (zero) in the second stream.
    This is demultiplexed in ScaledEmbedding in lm.py.
    
    Attributes:
        card: Text cardinality (vocab_size + 1 for special tokens).
             Used for multiplexing two tokens into one value.
        new_word: Token indicating a new word should start (default: 0)
        pad: Padding token for silence/continuation (default: 3)
        main: Token for main speaker turn start (default: 1)
        other: Token for other speaker turn start (default: 2)
        zero: Special value producing zero embedding (default: -1)
        ungenerated: Marker for ungenerated positions (default: -2)
    """

    card: int
    new_word: int = 0
    pad: int = 3
    main: int = 1
    other: int = 2
    zero: int = -1
    ungenerated: int = -2


@dataclass
class Entry:
    """
    One word to generate in the TTS pipeline.
    
    An Entry represents a single word (or pause) in the text to be
    synthesized. The state machine processes entries sequentially,
    feeding their tokens to the model when requested.
    
    ==========================================================================
    ENTRY TYPES
    ==========================================================================
    
    1. WORD ENTRY (tokens non-empty):
       - Contains tokenized word
       - Fed to model when it predicts new_word
       - May include speaker token at turn boundaries
    
    2. PAUSE ENTRY (tokens empty, padding > 0):
       - Represents a pause/break in speech
       - Created from <break time="Xs"/> SSML tags
       - Forces padding without feeding tokens
    
    ==========================================================================
    PADDING BEHAVIOR
    ==========================================================================
    
    The padding field controls minimum silence AFTER this word:
    - padding=0: Model can immediately request next word
    - padding=N: Model must pad N times before next word
    
    This prevents words from running together and allows
    control over speech rate.
    
    Attributes:
        tokens: List of token IDs for this word.
               Empty for pause-only entries.
        text: The word as a string (for logging/debugging).
        padding: Minimum padding steps after this word (default: 0).
                Higher values = slower speech.
        audio_tokens: Optional audio prefix to force into the model.
                     Used for voice cloning/continuation.
    """

    tokens: list[int]
    text: str
    padding: int = 0
    audio_tokens: mx.array | None = None


@dataclass
class State:
    """
    Current state of the TTS generation.
    
    The State tracks everything needed for the state machine to
    coordinate text feeding during generation. It's modified
    in-place as generation progresses.
    
    ==========================================================================
    STATE COMPONENTS
    ==========================================================================
    
    WORD QUEUE:
    - entries: Words waiting to be synthesized
    - queued: Token IDs for current word being fed
    - lookahead_queued: Tokens for lookahead stream (if enabled)
    
    PADDING CONTROL:
    - remaining_padding: How many more pads allowed before forced word
    - forced_padding: How many pads required before next word allowed
    
    GENERATION TRACKING:
    - end_step: Step when generation completed (all words consumed)
    - consumption_times: Step at which each entry was consumed
    - transcript: (word, step) pairs for timing information
    
    ==========================================================================
    PADDING LOGIC
    ==========================================================================
    
    At each step:
    1. If queued has tokens → must pad (feeding current word)
    2. If forced_padding > 0 → must pad (minimum silence)
    3. If remaining_padding <= 0 → must request new_word (max silence)
    4. Otherwise → model decides (pad or new_word)
    
    Attributes:
        entries: Queue of Entry objects waiting to be synthesized.
        remaining_padding: Pads remaining before forced new_word.
        forced_padding: Pads required before new_word allowed.
        queued: Token IDs for current word being fed to main stream.
        lookahead_queued: Token IDs for lookahead stream.
        end_step: Step when generation ended (None if ongoing).
        consumption_times: List of steps when entries were consumed.
        transcript: List of (word, step) tuples for timing.
    """

    entries: deque[Entry]
    remaining_padding: int
    forced_padding: int
    queued: deque[int] = field(default_factory=deque)
    lookahead_queued: deque[int] = field(default_factory=deque)
    end_step: int | None = None
    consumption_times: list[int] = field(default_factory=list)
    transcript: list[tuple[str, int]] = field(default_factory=list)

    def get_tokens_ahead(self, lookahead: int) -> list[int]:
        """
        Get tokens for the Nth word ahead in the queue.
        
        Used for the lookahead stream (second_stream_ahead) which
        provides the model with future context.
        
        Args:
            lookahead: How many words ahead to look (must be > 0)
        
        Returns:
            Token list for the Nth word ahead, or empty list if
            not enough words remain.
        """
        assert lookahead > 0
        for entry in self.entries:
            if entry.tokens:
                lookahead -= 1
                if lookahead == 0:
                    return entry.tokens
        return []


def _delayed(codes: mx.array, delays: list[int], fill_value: int) -> mx.array:
    """
    Apply acoustic delays to audio tokens.
    
    Shifts each codebook by its delay amount, filling the gap with
    fill_value. This aligns audio tokens with the delayed streams
    architecture used by Moshi.
    
    Args:
        codes: Audio tokens [K, T] where K is codebooks, T is timesteps
        delays: List of delay values for each codebook
        fill_value: Value to fill positions before the delay window
    
    Returns:
        Delayed tokens [K, T + max(delays)]
    
    Example:
        >>> codes = mx.array([[1, 2, 3], [4, 5, 6]])  # 2 codebooks, 3 steps
        >>> delays = [0, 1]
        >>> _delayed(codes, delays, -1)
        # [[1, 2, 3, -1],   # codebook 0: no delay
        #  [-1, 4, 5, 6]]   # codebook 1: delay=1
    """
    K, T = codes.shape
    out = mx.full((K, T + max(delays)), fill_value, dtype=mx.int64)
    for k, delay in enumerate(delays):
        out[k, delay : delay + T] = codes[k]
    return out


def _make_null(
    all_attributes: tp.Sequence[ConditionAttributes],
) -> list[ConditionAttributes]:
    """
    Create null (unconditional) versions of condition attributes for CFG.
    
    When using classifier-free guidance, we need both conditional and
    unconditional predictions. This function creates the unconditional
    version by dropping all conditioning information.
    
    Args:
        all_attributes: List of ConditionAttributes with conditioning info
    
    Returns:
        List of ConditionAttributes with all conditions dropped
    """
    return dropout_all_conditions(all_attributes)


@dataclass
class StateMachine:
    """
    State machine controlling text feeding during TTS generation.
    
    The StateMachine is the CORE of DSM-based TTS. It monitors the model's
    text predictions and decides what to feed as input at each step.
    
    ==========================================================================
    HOW IT WORKS
    ==========================================================================
    
    At each generation step:
    1. Model predicts a text token (new_word or pad)
    2. StateMachine.process() is called with the prediction
    3. State machine decides actual output based on:
       - Current state (queued tokens, padding counters)
       - Model's prediction
       - Constraints (max_padding, forced_padding)
    4. Returns the token to feed as input for next step
    
    ==========================================================================
    PADDING CONSTRAINTS
    ==========================================================================
    
    - max_padding: Maximum consecutive pads allowed
      If remaining_padding reaches 0, force new_word
      Prevents speech from being too slow
    
    - forced_padding: Minimum pads after each word
      Must pad this many times before allowing new_word
      Prevents speech from being too fast
    
    - initial_padding: Pads at generation start
      Prevents first word from being cut off
    
    ==========================================================================
    LOOKAHEAD (second_stream_ahead)
    ==========================================================================
    
    When second_stream_ahead > 0, the model receives two text streams:
    1. Main stream: Current word being spoken
    2. Lookahead stream: Word N positions ahead
    
    This gives the model future context for better prosody planning.
    The two streams are multiplexed into a single token value.
    
    Attributes:
        token_ids: TokenIds instance with special token values
        second_stream_ahead: Lookahead distance (0 = disabled)
        max_padding: Maximum consecutive pads (default: 6)
        initial_padding: Pads at start of generation (default: 2)
    """

    token_ids: TokenIds
    second_stream_ahead: int = 0
    max_padding: int = 6
    initial_padding: int = 2

    def new_state(self, entries: tp.Sequence[Entry]) -> State:
        """
        Create a new State for TTS generation.
        
        Initializes the state with the given entries and sets up
        padding counters for the start of generation.
        
        Args:
            entries: Sequence of Entry objects to synthesize
        
        Returns:
            Fresh State ready for generation
        """
        state = State(
            entries=deque(entries),
            lookahead_queued=deque(),
            remaining_padding=self.initial_padding,
            forced_padding=self.initial_padding,
        )
        return state

    def process(self, step: int, state: State, token: int) -> tuple[int, bool]:
        """
        Process model prediction and determine next input token.
        
        This is the MAIN LOGIC of the state machine. It takes the model's
        text prediction and decides what to actually feed as input.
        
        =======================================================================
        DECISION LOGIC
        =======================================================================
        
        1. OVERRIDE TO PAD if:
           - queued has tokens (still feeding current word)
           - forced_padding > 0 (minimum silence not met)
        
        2. OVERRIDE TO NEW_WORD if:
           - remaining_padding <= 0 (maximum silence exceeded)
        
        3. PROCESS NEW_WORD:
           - Pop next entry from queue
           - Queue its tokens for feeding
           - Reset padding counters
           - If entry is pause-only, treat as pad
        
        4. PROCESS PAD:
           - Decrement padding counters
           - If queued has tokens, feed one
           - Otherwise output pad token
        
        5. MULTIPLEX (if second_stream_ahead):
           - Combine main and lookahead tokens
           - output = (second + 1) * card + main
        
        =======================================================================
        
        Args:
            step: Current generation step index
            state: State to modify in-place
            token: Model's predicted token (new_word or pad)
        
        Returns:
            Tuple of (output_token, consumed_new_word):
            - output_token: Value to feed as next input
            - consumed_new_word: True if a new word was started
        """
        consumed_new_word = False
        
        # Sanitize token to valid values
        if token not in [self.token_ids.new_word, self.token_ids.pad]:
            token = self.token_ids.pad

        # =================================================================
        # OVERRIDE LOGIC
        # =================================================================
        if state.queued:
            # Still feeding current word's tokens → must pad
            token = self.token_ids.pad
        elif state.forced_padding > 0:
            # Minimum silence not met → must pad
            token = self.token_ids.pad
        elif state.remaining_padding <= 0:
            # Maximum silence exceeded → must request new word
            token = self.token_ids.new_word

        # =================================================================
        # PROCESS NEW_WORD
        # =================================================================
        if token == self.token_ids.new_word:
            if state.entries:
                entry = state.entries.popleft()
                state.consumption_times.append(step)
                
                if entry.tokens:
                    # Real word entry
                    consumed_new_word = True
                    state.transcript.append((entry.text, step))
                    
                    # Queue tokens for feeding
                    state.queued.extend(entry.tokens)
                    
                    # Queue lookahead tokens if enabled
                    if self.second_stream_ahead:
                        state.lookahead_queued.extend(
                            state.get_tokens_ahead(self.second_stream_ahead)
                        )
                    
                    # Reset max padding counter
                    state.remaining_padding = self.max_padding
                else:
                    # Pause-only entry (from <break> tag)
                    token = self.token_ids.pad
                
                # Set forced padding from entry
                state.forced_padding = entry.padding
            else:
                # No more entries
                token = self.token_ids.pad
                if self.second_stream_ahead and state.end_step is None:
                    token = self.token_ids.new_word
                
                # Mark end of generation
                if state.end_step is None:
                    state.end_step = step

        # =================================================================
        # DETERMINE OUTPUT TOKEN
        # =================================================================
        output: int | None = None
        
        if token == self.token_ids.pad:
            # Decrement padding counters
            if state.remaining_padding > 0:
                state.remaining_padding -= 1
            if state.forced_padding > 0:
                state.forced_padding -= 1
            
            if state.queued:
                # Feed next token from current word
                output = state.queued.popleft()
            else:
                # No tokens to feed → output pad
                output = self.token_ids.pad
                
        elif token == self.token_ids.new_word:
            output = self.token_ids.new_word
            
        elif token == self.token_ids.zero:
            output = token
            
        else:
            raise RuntimeError(f"Invalid token {token}")

        # =================================================================
        # MULTIPLEX WITH LOOKAHEAD STREAM
        # =================================================================
        if self.second_stream_ahead:
            second = -1  # Default: zero embedding
            
            if output == self.token_ids.new_word:
                # Put new_word on second stream, feed word token on main
                second = self.token_ids.new_word
                if state.queued:
                    output = state.queued.popleft()
                else:
                    output = self.token_ids.pad
            elif state.lookahead_queued:
                # Feed lookahead token on second stream
                second = state.lookahead_queued.popleft()
            
            # Multiplex: (second + 1) * card + main
            # +1 allows encoding -1 (zero) in second stream
            output = (second + 1) * self.token_ids.card + output

        assert output is not None
        return output, consumed_new_word


def script_to_entries(
    tokenizer: SentencePieceProcessor,
    token_ids: TokenIds,
    frame_rate: float,
    script: tp.Sequence[str],
    multi_speaker: bool = True,
    padding_between: int = 0,
) -> list[Entry]:
    """
    Convert a text script into Entry objects for TTS generation.
    
    This is the TEXT PREPROCESSING step that transforms human-readable text
    into the structured format consumed by the TTS state machine.
    
    ==========================================================================
    PROCESSING PIPELINE
    ==========================================================================
    
    1. CHARACTER NORMALIZATION:
       - Replace curly quotes with straight quotes
       - Remove colons (replaced with spaces)
       - Remove parentheses
    
    2. TOKENIZATION:
       - Split text into words
       - Tokenize each word with SentencePiece
       - Add speaker tokens at turn boundaries
    
    3. SSML PARSING:
       - Parse <break time="Xs"/> tags for pauses
       - Convert time to frame count using frame_rate
       - Create pause-only Entry objects
    
    ==========================================================================
    MULTI-SPEAKER HANDLING
    ==========================================================================
    
    When multi_speaker=True:
    - Each element in script represents a speaker turn
    - Odd indices = main speaker, even indices = other speaker
    - Speaker tokens (main/other) inserted at turn boundaries
    - Empty first turn starts with "other" speaker
    
    ==========================================================================
    PADDING BETWEEN WORDS
    ==========================================================================
    
    The padding_between parameter adds forced silence between words:
    - padding = max(0, padding_between + len(tokens) - 1)
    - Higher values = slower, more articulated speech
    - Useful for clarity in complex words
    
    Args:
        tokenizer: SentencePiece tokenizer for text encoding.
        token_ids: TokenIds instance with special token values.
        frame_rate: Audio codec frame rate (frames per second).
                   Used to convert break durations to frame counts.
        script: List of text turns. Each element is one speaker's turn.
               Alternates between main and other speaker.
        multi_speaker: If True, insert speaker tokens at turn boundaries.
        padding_between: Base padding to add between words (default: 0).
    
    Returns:
        List of Entry objects ready for the TTS state machine.
    
    Example:
        >>> script = ["Hello there!", "Hi, how are you?"]
        >>> entries = script_to_entries(tokenizer, token_ids, 12.5, script)
        # Returns entries for: [main]Hello, there!, [other]Hi,, how, are, you?
        
        >>> script = ["Wait <break time=\"1s\"/> okay"]
        >>> entries = script_to_entries(tokenizer, token_ids, 12.5, script)
        # Returns entries for: [main]Wait, [pause 12 frames], okay
    """
    speaker_tokens = [token_ids.main, token_ids.other]
    last_speaker = None
    entries = []

    # break is indicated as e.g. <break time="3s"/>
    event_re = re.compile(r"(?:<break\s+time=\"([0-9]+(?:.[0-9]*)?)s\"\s*/?>)|(?:\s+)")

    def _add_entry(idx: int, word: str):
        nonlocal first_content, last_speaker
        assert " " not in word
        assert word
        tokens = tokenizer.encode(word)  # type: ignore
        if first_content:
            speaker = idx % len(speaker_tokens)
            if multi_speaker and last_speaker != speaker:
                last_speaker = speaker
                tokens.insert(0, speaker_tokens[speaker])
            first_content = False
        padding = 0
        if padding_between > 0:
            padding = max(0, padding_between + len(tokens) - 1)
        entries.append(Entry(tokens=tokens, text=word, padding=padding))

    for idx, line in enumerate(script):
        first_content = True
        line = line.replace("’", "'")
        line = line.replace(":", " ")
        line = line.replace("(", "")
        line = line.replace(")", "")
        while line:
            match = event_re.search(line)
            if match is None:
                break
            word = line[: match.start()]
            line = line[match.end() :]
            if word:
                _add_entry(idx, word)
            if match.group(1):
                break_duration = float(match.group(1))
                padding = int(round(break_duration * frame_rate))
                entry = Entry(tokens=[], text="", padding=padding)
                entries.append(entry)
        if line:
            _add_entry(idx, line)
    return entries


@dataclass
class TTSResult:
    """
    Output of TTS generation containing audio tokens and timing metadata.
    
    TTSResult encapsulates everything produced by a TTS generation run,
    including the generated audio tokens and detailed timing information
    for synchronization and debugging.
    
    ==========================================================================
    FRAME FORMAT
    ==========================================================================
    
    Each frame in `frames` has shape [B, 1 + Q, 1] where:
    - B = batch size
    - 1 = text token (first position)
    - Q = audio codebook tokens
    - 1 = single timestep
    
    IMPORTANT: Acoustic delays are ALREADY CORRECTED in the frames.
    The frames can be directly fed to Mimi decoder without additional
    delay handling.
    
    ==========================================================================
    TIMING INFORMATION
    ==========================================================================
    
    Three types of timing data are provided:
    
    1. end_steps: When generation completed for each batch item
       - None if text wasn't fully synthesized within budget
       - Used to trim output to valid portion
    
    2. all_consumption_times: When each Entry was consumed
       - Useful for debugging state machine behavior
       - Maps Entry index to generation step
    
    3. all_transcripts: Word-level timing
       - (word, step) pairs for each word
       - Divide step by frame_rate for timestamp in seconds
       - Enables subtitle generation, lip sync, etc.
    
    ==========================================================================
    USAGE
    ==========================================================================
    
        result = tts_model.generate(entries, attributes)
        
        # Decode audio
        for frame in result.frames:
            audio_chunk = mimi.decode_step(frame[:, 1:, :])  # Skip text token
        
        # Get word timings
        for word, step in result.all_transcripts[0]:
            timestamp = step / frame_rate
            print(f"{word} at {timestamp:.2f}s")
    
    Attributes:
        frames: List of token tensors [B, 1+Q, 1] for each generation step.
               Acoustic delays already corrected.
        logged_text_tokens: Debug info - list of (predicted, actual_input) pairs.
                           Shows state machine decisions at each step.
        end_steps: Step when generation ended for each batch item.
                  None if text wasn't fully consumed within max_gen_length.
        all_consumption_times: Per-batch list of steps when Entries were consumed.
        all_transcripts: Per-batch list of (word, step) timing pairs.
    """

    frames: list[mx.array]
    logged_text_tokens: list[list[tuple[int, int]]]
    end_steps: list[int | None]
    all_consumption_times: list[list[int]]
    all_transcripts: list[list[tuple[str, int]]]


@dataclass
class TTSModel:
    """
    High-level TTS interface combining LM, Mimi codec, and tokenizer.
    
    TTSModel is the MAIN ENTRY POINT for text-to-speech synthesis using
    Moshi's Delayed Streams Modeling architecture. It orchestrates:
    - Text preprocessing (script → Entry objects)
    - Voice conditioning (speaker embeddings)
    - Generation (LM + state machine)
    - Audio decoding (Mimi codec)
    
    ==========================================================================
    ARCHITECTURE OVERVIEW
    ==========================================================================
    
    TTSModel wraps three core components:
    
    1. LM (Language Model):
       - Temporal Transformer (Helium) for sequence modeling
       - DepFormer for audio token generation
       - Handles text + audio co-generation
    
    2. Mimi (Neural Audio Codec):
       - Encodes audio → tokens (for prefixes/conditioning)
       - Decodes tokens → audio (for output)
       - 8 RVQ codebook levels
    
    3. Tokenizer (SentencePiece):
       - Converts text to token IDs
       - Handles subword tokenization
    
    ==========================================================================
    GENERATION FLOW
    ==========================================================================
    
    1. prepare_script(): Convert text to Entry objects
    2. get_voice_path(): Load voice embedding file
    3. make_condition_attributes(): Create conditioning tensors
    4. generate(): Run TTS generation loop
    5. Decode frames with Mimi
    
    ==========================================================================
    VOICE CONDITIONING
    ==========================================================================
    
    Speaker identity is controlled via cross-attention conditioning:
    - Voice embeddings stored in .safetensors files
    - Up to max_speakers (5) voices per generation
    - Embeddings provide speaker characteristics, style, prosody
    
    ==========================================================================
    CLASSIFIER-FREE GUIDANCE
    ==========================================================================
    
    Two CFG modes supported:
    
    1. Direct CFG (cfg_coef != 1.0):
       - Computes conditional and unconditional logits
       - Interpolates: logits = cfg_coef * cond - (cfg_coef - 1) * uncond
       - NOT supported by CFG-distilled models
    
    2. CFG Distillation:
       - Model trained with CFG baked in
       - Pass cfg_coef to make_condition_attributes()
       - Valid values: typically 1.0 to 4.0 in 0.5 increments
    
    ==========================================================================
    USAGE EXAMPLE
    ==========================================================================
    
        # Load model (use from_checkpoint_info in practice)
        tts = TTSModel(lm, mimi, tokenizer, ...)
        
        # Prepare text
        script = ["Hello, how are you?", "I'm doing great!"]
        entries = tts.prepare_script(script)
        
        # Get voice conditioning
        voice_path = tts.get_voice_path("alice")
        attributes = tts.make_condition_attributes([voice_path])
        
        # Generate
        result = tts.generate([entries], [attributes])
        
        # Decode audio
        audio_chunks = []
        for frame in result.frames:
            chunk = mimi.decode_step(frame[:, 1:, :])
            audio_chunks.append(chunk)
    
    Attributes:
        lm: Trained Delayed Streams language model.
        mimi: Mimi neural audio codec for encoding/decoding.
        tokenizer: SentencePiece tokenizer for text.
        voice_suffix: File suffix for voice embeddings (includes model signature).
        voice_repo: HuggingFace repo for voice files.
        machine: StateMachine controlling text feeding.
        delay_steps: Delay between text and audio in generation steps.
        max_speakers: Maximum speakers for cross-attention (default: 5).
        temp: Sampling temperature for text and audio (default: 0.6).
        cfg_coef: CFG coefficient (default: 1.0 = no CFG).
        final_padding: Steps to generate after last word (default: 4).
        n_q: Number of audio codebooks to generate (default: 32).
        max_gen_length: Maximum generation steps (default: 30000).
        padding_bonus: Bonus for pad logits, positive = slower speech (default: 0.0).
    """

    lm: Lm
    mimi: Mimi
    tokenizer: SentencePieceProcessor

    voice_suffix: str
    voice_repo: str

    machine: StateMachine
    delay_steps: int
    max_speakers: int = 5

    # The following params can be overriden to customize generation.
    temp: float = 0.6
    cfg_coef: float = 1.0
    final_padding: int = 4
    n_q: int = 32
    max_gen_length: int = 30000
    padding_bonus: float = 0.0

    def __init__(
        self,
        lm: Lm,
        mimi: Mimi,
        tokenizer: SentencePieceProcessor,
        temp: float = 0.6,
        cfg_coef: float = 1.0,
        final_padding: int = 4,
        n_q: int = 32,
        max_gen_length: int = 30000,
        padding_bonus: float = 0.0,
        initial_padding: int = 2,
        max_padding: int = 8,
        voice_repo: str = DEFAULT_DSM_TTS_VOICE_REPO,
        raw_config: dict = {},
    ):
        """
        Initialize TTSModel with all required components.
        
        This constructor sets up the TTS pipeline by:
        1. Storing model components (LM, Mimi, tokenizer)
        2. Configuring generation parameters
        3. Creating the state machine with timing constraints
        4. Extracting model-specific settings from raw_config
        
        NOTE: End users should use from_checkpoint_info() instead of
        calling this constructor directly.
        
        Args:
            lm: Trained Delayed Streams language model.
            mimi: Mimi neural audio codec.
            tokenizer: SentencePiece text tokenizer.
            temp: Sampling temperature (default: 0.6).
            cfg_coef: CFG coefficient (default: 1.0 = disabled).
            final_padding: Steps after last word (default: 4).
            n_q: Audio codebooks to generate (default: 32).
            max_gen_length: Maximum steps (default: 30000).
            padding_bonus: Pad logit bonus (default: 0.0).
            initial_padding: Pads at generation start (default: 2).
            max_padding: Max consecutive pads (default: 8).
            voice_repo: HuggingFace repo for voices.
            raw_config: Model config dict with model_id and tts_config.
        """
        self.lm = lm
        self.mimi = mimi
        self.tokenizer = tokenizer
        self.temp = temp
        self.cfg_coef = cfg_coef
        self.final_padding = final_padding
        self.n_q = n_q
        self.max_gen_length = max_gen_length
        self.padding_bonus = padding_bonus
        model_id = raw_config["model_id"]
        self.voice_suffix = f".{model_id['sig']}@{model_id['epoch']}.safetensors"
        self.voice_repo = voice_repo

        token_ids = TokenIds(lm.cfg.text_out_vocab_size + 1)
        tts_config = raw_config["tts_config"]
        self.delay_steps = int(tts_config["audio_delay"] * mimi.frame_rate)
        second_stream_ahead = tts_config.get("second_stream_ahead", 0)

        self.machine = StateMachine(
            token_ids=token_ids,
            second_stream_ahead=second_stream_ahead,
            max_padding=max_padding,
            initial_padding=initial_padding,
        )

    @cached_property
    def valid_cfg_conditionings(self) -> set[float]:
        """
        Get valid CFG coefficient values for CFG-distilled models.
        
        For models trained with CFG distillation, only specific CFG
        values are valid (typically 1.0 to 4.0 in 0.5 increments).
        This property extracts valid values from the LutConditioner.
        
        Returns:
            Set of valid CFG coefficients, or empty set if model
            doesn't use CFG distillation.
        """
        valid_cfg_conditionings = set()
        if (
            self.lm.condition_provider is not None
            and "cfg" in self.lm.condition_provider.conditioners
        ):
            cfg_conditioner = self.lm.condition_provider.conditioners["cfg"]
            assert isinstance(cfg_conditioner, LutConditioner)
            assert cfg_conditioner.possible_values is not None
            valid_cfg_conditionings = set(
                float(x) for x in cfg_conditioner.possible_values
            )
        return valid_cfg_conditionings

    @cached_property
    def multi_speaker(self) -> bool:
        """
        Check if model supports multiple speakers.
        
        Multi-speaker support is determined by the presence of a
        'speaker_wavs' conditioner in the condition provider.
        
        Returns:
            True if model supports voice conditioning, False otherwise.
        """
        if self.lm.condition_provider is None:
            return False
        return "speaker_wavs" in self.lm.condition_provider.conditioners

    def prepare_script(
        self, script: tp.Sequence[str], padding_between: int = 0
    ) -> list[Entry]:
        """
        Convert text script to Entry objects for generation.
        
        Convenience wrapper around script_to_entries() that uses
        this model's tokenizer, token_ids, and frame_rate.
        
        Args:
            script: List of text turns (alternating speakers).
            padding_between: Extra padding between words (default: 0).
        
        Returns:
            List of Entry objects ready for generate().
        
        Example:
            >>> entries = tts.prepare_script(["Hello!", "Hi there!"])
        """
        return script_to_entries(
            self.tokenizer,
            self.machine.token_ids,
            self.mimi.frame_rate,
            script,
            multi_speaker=self.multi_speaker,
            padding_between=padding_between,
        )

    def generate(
        self,
        all_entries: tp.Sequence[tp.Sequence[Entry]],
        attributes: tp.Sequence[tp.Any],
        prefixes: list[mx.array] | None = None,
        cfg_is_no_prefix: bool = True,
        cfg_is_no_text: bool = True,
        on_audio_hook: tp.Optional[tp.Callable[[mx.array], None]] = None,
        on_text_hook: tp.Optional[tp.Callable[[mx.array], None]] = None,
        on_frame: tp.Optional[tp.Callable[[mx.array], None]] = None,
    ) -> TTSResult:
        """
        Synthesize text to audio using Delayed Streams Modeling.
        
        This is the MAIN GENERATION METHOD that runs the TTS loop:
        1. Initialize state machines for each batch item
        2. Set up conditioning (voice embeddings, CFG)
        3. Run generation loop until all text consumed or max_gen_length
        4. Return frames and timing metadata
        
        =======================================================================
        GENERATION LOOP
        =======================================================================
        
        At each step:
        1. LmGen.step() generates text and audio tokens
        2. _on_text_hook processes text via state machine
        3. _on_audio_hook handles delays and prefixes
        4. Valid frames are collected for output
        
        Generation stops when:
        - All state machines reach end_step, OR
        - max_gen_length is reached
        
        =======================================================================
        CFG HANDLING
        =======================================================================
        
        If cfg_coef != 1.0 and model supports direct CFG:
        - Attributes are duplicated with null versions
        - LmGen computes both conditional and unconditional logits
        - Final logits = uncond + cfg_coef * (cond - uncond)
        
        =======================================================================
        PREFIX HANDLING
        =======================================================================
        
        Prefixes allow voice cloning / continuation:
        - Audio tokens from prefix are forced into generation
        - Text tokens from prefix bypass state machine
        - cfg_is_no_prefix masks prefix for null logits
        
        =======================================================================
        HOOKS
        =======================================================================
        
        Three callback hooks for streaming/monitoring:
        - on_audio_hook: Called after audio token generation
        - on_text_hook: Called after text token generation
        - on_frame: Called when a complete frame is ready
        
        Args:
            all_entries: Batch of Entry sequences from prepare_script().
            attributes: ConditionAttributes for each batch item.
            prefixes: Optional audio prefixes for continuation [K, T].
            cfg_is_no_prefix: Mask prefix for CFG null logits (default: True).
            cfg_is_no_text: Exclude text for CFG null logits (default: True).
            on_audio_hook: Callback for audio tokens.
            on_text_hook: Callback for text tokens.
            on_frame: Callback for complete frames.
        
        Returns:
            TTSResult with frames and timing metadata.
        """

        # TODO(laurent):
        # Re-enable the padding bonus.
        # def _main_wrapper(*args, **kwargs):
        #     transformer_out, text_logits = original(*args, **kwargs)
        #     if self.padding_bonus:
        #         text_logits[..., self.machine.token_ids.pad] += self.padding_bonus
        #     return transformer_out, text_logits

        for c in self.lm.transformer_cache:
            c.reset()
        for c in self.lm.depformer_cache:
            c.reset()
        self.mimi.reset_all()

        if self.cfg_coef != 1.0:
            if self.valid_cfg_conditionings:
                raise ValueError(
                    "This model does not support direct CFG, but was trained with "
                    "CFG distillation. Pass instead `cfg_coef` to `make_condition_attributes`."
                )
            nulled = _make_null(attributes)
            attributes = list(attributes) + nulled

        assert self.lm.condition_provider is not None
        batch_size = len(all_entries)
        ct_list = []
        cross_attention_src_list = []

        for _attr in attributes:
            ct = None
            cross_attention_src = None
            for _key, _value in _attr.text.items():
                _ct = self.lm.condition_provider.condition_tensor(_key, _value)
                tensor = _ct.tensor.squeeze(0)
                ct = tensor if ct is None else ct + tensor
            ct_list.append(ct)
            for _key, _value in _attr.tensor.items():
                _conditioner = self.lm.condition_provider.conditioners[_key]
                _ca_src = _conditioner.condition(_value)
                if cross_attention_src is None:
                    cross_attention_src = _ca_src
                else:
                    raise ValueError("multiple cross-attention conditioners")
            cross_attention_src_list.append(cross_attention_src)
        cross_attention_src = mx.concatenate(cross_attention_src_list, axis=0)
        ct = ConditionTensor(mx.stack(ct_list, axis=0))

        states = []

        for entries in all_entries:
            state = self.machine.new_state(entries)
            states.append(state)

        cfg_is_masked_until = None
        text_prefixes = None
        audio_prefixes = None
        if prefixes is not None:
            assert len(all_entries) == len(prefixes), (
                f"Not enough prefixes, expected {len(all_entries)}."
            )
            if cfg_is_no_prefix:
                cfg_is_masked_until = []
            text_prefixes = []
            audio_prefixes = []
            for prefix in prefixes:
                if cfg_is_masked_until is not None:
                    cfg_is_masked_until.append(prefix.shape[-1] + self.delay_steps)
                K, _ = prefix.shape
                assert K == self.lm.num_codebooks
                text_prefixes.append(deque(prefix[0].tolist()))
                delays = [
                    d + self.delay_steps for d in self.lm.delays[self.lm.audio_offset :]
                ]
                delayed = _delayed(
                    prefix[self.lm.audio_offset :],
                    delays,
                    self.machine.token_ids.ungenerated,
                )
                audio_prefixes.append(deque(delayed.T))

        def _on_audio_hook(audio_tokens):
            delays = self.lm.delays
            ungenerated = self.machine.token_ids.ungenerated
            for q in range(audio_tokens.shape[1]):
                delay = delays[q]
                if offset < delay + self.delay_steps:
                    audio_tokens[:, q] = self.machine.token_ids.zero
            if audio_prefixes is not None:
                for b, audio_prefix in enumerate(audio_prefixes):
                    if audio_prefix:
                        audio_codes = audio_prefix.popleft()
                        mask = audio_codes != ungenerated
                        audio_tokens[b] = mx.where(mask, audio_codes, audio_tokens[b])
            if on_audio_hook is not None:
                on_audio_hook(audio_tokens)

        def _on_text_hook(text_tokens):
            tokens = text_tokens.tolist()
            out_tokens = []
            for b, (token, state, logged) in enumerate(
                zip(tokens, states, logged_text_tokens)
            ):
                if text_prefixes is not None and text_prefixes[b]:
                    out_token = text_prefixes[b].popleft()
                else:
                    out_token, _ = self.machine.process(offset, state, token[0])
                out_tokens.append(out_token)
                logged.append((token, out_token))
            text_tokens[:] = mx.array(out_tokens, dtype=mx.int64)[:, None]
            if on_text_hook is not None:
                on_text_hook(text_tokens)

        lm_gen = LmGen(
            self.lm,
            max_steps=self.max_gen_length,
            text_sampler=Sampler(temp=self.temp),
            audio_sampler=Sampler(temp=self.temp),
            batch_size=batch_size,
            cfg_coef=self.cfg_coef,
            on_text_hook=_on_text_hook,
            on_audio_hook=_on_audio_hook,
            # TODO(laurent):
            # cfg_is_masked_until=cfg_is_masked_until,
            # cfg_is_no_text=cfg_is_no_text,
        )

        logged_text_tokens = [[] for _ in states]
        frames: list[mx.array] = []

        for offset in range(self.max_gen_length):
            if all(state.end_step is not None for state in states):
                max_end_step = max(state.end_step for state in states)
                if offset >= max_end_step + self.delay_steps + self.final_padding:
                    break
            missing = self.lm.n_q - self.lm.dep_q
            input_tokens = (
                mx.ones((len(states), missing), dtype=mx.int64)
                * self.machine.token_ids.zero
            )
            lm_gen.step(input_tokens, ct=ct, cross_attention_src=cross_attention_src)
            frame = lm_gen.last_audio_tokens()

            if frame is not None and (frame != self.machine.token_ids.zero).all():
                frames.append(mx.array(frame)[:, :, None])

                if on_frame is not None:
                    on_frame(frame)

        return TTSResult(
            frames,
            logged_text_tokens,
            [state.end_step for state in states],
            [state.consumption_times for state in states],
            [state.transcript for state in states],
        )

    def get_voice_path(self, voice_name: str) -> Path:
        """
        Get local path to a voice embedding file.
        
        Fetches voice embeddings from HuggingFace if not cached locally.
        Voice files contain pre-computed speaker embeddings for conditioning.
        
        The voice file path is constructed as:
            {voice_name}{voice_suffix}
        
        Where voice_suffix includes model signature and epoch.
        
        Args:
            voice_name: Name of the voice (e.g., "alice", "bob").
                       Can also use "hf://REPO/PATH" syntax for
                       voices from other repositories.
        
        Returns:
            Path to the local voice embedding file (.safetensors).
        
        Example:
            >>> path = tts.get_voice_path("alice")
            >>> # Returns: ~/.cache/huggingface/.../alice.sig@epoch.safetensors
        """
        file = hf_get(
            voice_name + self.voice_suffix,
            self.voice_repo,
            check_local_file_exists=True,
        )
        return Path(file)

    def make_condition_attributes(
        self, voices: list[Path], cfg_coef: float | None = None
    ):
        """
        Create ConditionAttributes from voice embedding files.
        
        Loads voice embeddings and packages them into the format
        expected by the LM's condition provider for cross-attention.
        
        =======================================================================
        VOICE TENSOR FORMAT
        =======================================================================
        
        Voice embeddings are loaded from .safetensors files containing
        'speaker_wavs' key. The tensor is reshaped for cross-attention:
        
        Input:  [1, T, D] per voice file
        Output: [1, max_speakers * T, D] concatenated
        
        A mask indicates which positions are valid (have voice data).
        
        =======================================================================
        CFG DISTILLATION
        =======================================================================
        
        For CFG-distilled models, pass cfg_coef here instead of using
        it directly in generation. Valid values are model-specific
        (check valid_cfg_conditionings property).
        
        Args:
            voices: List of paths to voice embedding files.
                   Up to max_speakers (5) voices supported.
            cfg_coef: CFG coefficient for distilled models.
                     Must be in valid_cfg_conditionings if provided.
        
        Returns:
            ConditionAttributes with voice tensors and text conditions.
        
        Raises:
            ValueError: If cfg_coef not in valid_cfg_conditionings.
        
        Example:
            >>> voice_path = tts.get_voice_path("alice")
            >>> attrs = tts.make_condition_attributes([voice_path], cfg_coef=2.0)
        """
        if voices:
            voice_tensor = None
            mask = None
            for idx in range(5):
                if idx < len(voices):
                    emb: mx.array = mx.load(str(voices[idx]))["speaker_wavs"]
                    assert emb.ndim == 3
                    if voice_tensor is None:
                        voice_tensor = mx.zeros(
                            (1, self.max_speakers, emb.shape[2], emb.shape[1])
                        )
                    if mask is None:
                        mask = mx.zeros(
                            (1, self.max_speakers, emb.shape[2]), dtype=mx.uint8
                        )
                    voice_tensor[:, idx, :, :] = emb.swapaxes(1, 2)
                    mask[:, idx, :] = True
            assert voice_tensor is not None
            assert mask is not None
            voice_tensor = voice_tensor.reshape(1, -1, voice_tensor.shape[-1])
            mask = mask.reshape(1, -1)
            tensors = {"speaker_wavs": TensorCondition(voice_tensor, mask)}
        else:
            tensors = {}
        text: dict[str, str | None] = {"control": "ok"}
        if cfg_coef is None:
            text["cfg"] = None
        else:
            if cfg_coef in self.valid_cfg_conditionings:
                text["cfg"] = format(cfg_coef, ".1f")
            else:
                valids = ", ".join(str(x) for x in self.valid_cfg_conditionings)
                raise ValueError(
                    f"Unsupported value for cfg_coef, valid values are {valids}."
                )

        return ConditionAttributes(text=text, tensor=tensors)

    def get_prefix(self, audio_path: Path | str) -> mx.array:
        """
        Encode audio file as prefix tokens for voice cloning/continuation.
        
        Loads an audio file, encodes it with Mimi, and formats it as
        a prefix that can be passed to generate(). This enables:
        - Voice cloning from audio samples
        - Continuing from existing audio
        - Style transfer from reference audio
        
        The prefix includes:
        - Row 0: Null text tokens (zero embedding)
        - Rows 1-K: Audio codebook tokens from Mimi encoding
        
        Args:
            audio_path: Path to audio file (any format supported by sphn).
        
        Returns:
            Prefix tensor [K, T] where K = num_codebooks, T = timesteps.
            Ready to pass to generate(prefixes=[prefix]).
        
        Example:
            >>> prefix = tts.get_prefix("reference_audio.wav")
            >>> result = tts.generate(entries, attrs, prefixes=[prefix])
        """
        wav, _ = sphn.read(audio_path, sample_rate=self.mimi.sample_rate)
        prefix = self.mimi.encode(mx.array(wav)[None])[0, :, :-2]
        null_text = mx.ones_like(prefix[:1]) * self.machine.token_ids.zero
        prefix = mx.concat([null_text, prefix], axis=0)
        return prefix
