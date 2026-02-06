# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Web-Based Local Inference Server for Moshi
==========================================

This module implements a web server that enables real-time voice conversations
with the Moshi model through a browser interface. It uses a multi-process
architecture to separate the concerns of:

1. Web Server Process: Handles WebSocket connections, audio encoding/decoding,
   and serves the static web interface
2. Model Server Process: Runs the language model inference on the GPU

The two processes communicate via multiprocessing queues, allowing the model
to run continuously while the web server handles network I/O asynchronously.

Architecture Overview:
---------------------
```
Browser <--WebSocket--> Web Server Process <--Queue--> Model Server Process
                              |                              |
                        Audio Codec (Mimi)              Language Model
                        Opus Encoding/Decoding          Token Generation
```

Key Features:
- Real-time bidirectional audio streaming via WebSocket
- Opus audio compression for efficient network transfer
- Support for quantized models (4-bit and 8-bit)
- Optional SSL/HTTPS support
- Automatic browser launch
- Static web interface serving

Usage:
------
    python -m moshi_mlx.local_web --hf-repo kyutai/moshiko-mlx-q8

The server will start and open a browser window to the local interface.
"""

import argparse
import asyncio
import json
import queue
import os
import tarfile
import time
import sys
import numpy as np
import multiprocessing
from pathlib import Path
import sentencepiece
from enum import Enum
import typing as tp
import sphn
import aiohttp
from aiohttp import web
import webbrowser

import mlx.core as mx
import mlx.nn as nn

import rustymimi
from moshi_mlx import models, utils

import huggingface_hub

# Audio configuration constants for the Mimi codec
SAMPLE_RATE = 24000  # 24kHz sample rate for high-quality speech
FRAME_SIZE = 1920    # 80ms frames (1920 samples at 24kHz)
CHANNELS = 1         # Mono audio


def colorize(text, color):
    """
    Apply ANSI color codes to text for terminal output.
    
    Args:
        text: The string to colorize
        color: ANSI color code (e.g., "1;31" for bold red)
    
    Returns:
        Text wrapped with ANSI escape sequences
    """
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def log(level: str, msg: str):
    """
    Print a formatted log message with colorized prefix.
    
    Log levels and their colors:
    - "info": Blue prefix [Info]
    - "warning": Red prefix [Warn]
    - "error": Red prefix [Err ]
    
    Args:
        level: Log severity ("info", "warning", or "error")
        msg: The message to log
    
    Raises:
        ValueError: If an unknown log level is provided
    """
    if level == "warning":
        prefix = colorize("[Warn]", "1;31")
    elif level == "info":
        prefix = colorize("[Info]", "1;34")
    elif level == "error":
        prefix = colorize("[Err ]", "1;31")
    else:
        raise ValueError(f"Unknown level {level}")
    print(prefix + " " + msg)


def hf_hub_download(repo, path: str) -> str:
    """
    Download a file from a HuggingFace Hub repository.
    
    Wrapper around huggingface_hub.hf_hub_download that validates
    the repository argument is provided.
    
    Args:
        repo: The HuggingFace repository ID (e.g., "kyutai/moshiko-mlx-q8")
        path: The path to the file within the repository
    
    Returns:
        Local path to the downloaded file
    
    Raises:
        ValueError: If repo is None or empty
    """
    if repo is None or repo == "":
        raise ValueError(f"the --hf-repo flag is required to retrieve {path}")
    return huggingface_hub.hf_hub_download(repo, path)


class Stats:
    """
    Container for timing statistics during inference.
    
    Tracks various timing metrics to help diagnose performance issues
    and measure latency in the audio processing pipeline.
    
    Attributes:
        send_times: Timestamps when audio was sent to the model
        model_times: Tuples of (start, end) times for model inference
        recv_times: Timestamps when audio was received from the model
    """
    send_times: tp.List[float] = []
    model_times: tp.List[tp.Tuple[float, float]] = []
    recv_times: tp.List[float] = []


class PrinterType(Enum):
    """
    Enumeration of message types for the inter-process printer queue.
    
    Used to communicate different types of events from worker processes
    to the main process for display.
    
    Values:
        TOKEN: A text token to display
        PENDING: Show the pending/processing indicator
        INFO: An informational log message
        WARNING: A warning log message
        ERROR: An error log message
        LAG: Audio processing is falling behind real-time
        HEADER: Print the output header
        EVENT: A timing event for profiling
        QSIZE: Queue size update for monitoring
    """
    TOKEN = 1
    PENDING = 2
    INFO = 3
    WARNING = 4
    ERROR = 5
    LAG = 6
    HEADER = 7
    EVENT = 8
    QSIZE = 9


def full_warmup(audio_tokenizer, client_to_server, server_to_client, max_delay: int):
    """
    Perform a full warmup of the audio tokenizer and model pipeline.
    
    This function sends silent audio frames through the entire pipeline to:
    1. Initialize the audio tokenizer's internal state
    2. Fill the model's delay buffer with initial tokens
    3. Ensure all components are ready for real-time processing
    
    The warmup is essential because the model uses delayed audio tokens,
    meaning it needs several frames of context before it can start
    generating meaningful output.
    
    Args:
        audio_tokenizer: The Mimi audio tokenizer instance
        client_to_server: Queue for sending audio tokens to the model
        server_to_client: Queue for receiving audio tokens from the model
        max_delay: Maximum delay in frames (determines warmup iterations)
    """
    for i in range(4):
        # Create a silent audio frame (1920 samples of zeros)
        pcm_data = np.array([0.0] * 1920).astype(np.float32)
        audio_tokenizer.encode(pcm_data)
        
        # Wait for the encoder to produce tokens
        while True:
            time.sleep(0.01)
            data = audio_tokenizer.get_encoded()
            if data is not None:
                break
        
        # Send tokens to the model server
        client_to_server.put_nowait(data)
        
        # Skip receiving for initial frames (filling the delay buffer)
        if i < max_delay:
            continue
        
        # Receive and decode the model's audio output
        while True:
            kind, data = server_to_client.get()
            if kind == 0:  # Audio tokens
                audio_tokenizer.decode(data)
                break
        
        # Wait for the decoder to produce PCM
        while True:
            time.sleep(0.01)
            data = audio_tokenizer.get_decoded()
            if data is not None:
                break


def hf_get(filename: str) -> str:
    """
    Resolve a filename that may be a HuggingFace URL or local path.
    
    Supports the "hf://" URL scheme for downloading files from HuggingFace Hub.
    Format: hf://owner/repo/path/to/file
    
    Args:
        filename: Either a local path or an hf:// URL
    
    Returns:
        Local path to the file (downloaded if necessary)
    
    Example:
        >>> hf_get("hf://kyutai/moshiko-mlx-q8/model.safetensors")
        '/path/to/cached/model.safetensors'
        >>> hf_get("./local/model.safetensors")
        './local/model.safetensors'
    """
    if filename.startswith("hf://"):
        parts = filename[5:].split("/")
        repo_name = parts[0] + "/" + parts[1]
        filename = "/".join(parts[2:])
        log("info", f"retrieving {filename} from hf repo {repo_name}")
        return hf_hub_download(repo_name, filename)
    else:
        return filename


def model_server(client_to_server, server_to_client, lm_config, args):
    """
    Model server process that runs the language model inference.
    
    This function runs in a separate process and handles:
    1. Loading the language model and text tokenizer
    2. Applying quantization if requested
    3. Running the main inference loop
    
    The inference loop continuously:
    - Receives audio tokens from the client via the input queue
    - Runs the model to generate text and audio tokens
    - Sends the generated tokens back via the output queue
    
    Communication Protocol:
    - Input: Audio tokens as numpy arrays from the audio tokenizer
    - Output: Tuples of (kind, data) where:
        - kind=0: Audio tokens (numpy array)
        - kind=1: Text string
    
    Args:
        client_to_server: Queue receiving audio tokens from the web server
        server_to_client: Queue sending generated tokens to the web server
        lm_config: Language model configuration (dict or LmConfig)
        args: Command-line arguments with model paths and settings
    
    Returns:
        None: This function runs indefinitely until interrupted
        
    Raises:
        ValueError: If an invalid quantization value is provided
        KeyboardInterrupt: When the process is terminated by user signal
        
    Note:
        This function is designed to run in a separate process and communicates
        with the web server process through multiprocessing queues. It handles
        the core AI inference workload while the web server manages network I/O.
    """
    # =========================================================================
    # Debug Mode Setup
    # =========================================================================
    if args.debug_model:
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))
        log("info", "[SERVER] Debugger listening on port 5678. Waiting for client...")
        debugpy.wait_for_client()
        log("info", "[SERVER] Debugger attached!")

    # =========================================================================
    # Model and Tokenizer Loading
    # =========================================================================
    # This section handles the initialization of the AI model and text tokenizer.
    # It supports multiple model formats (full precision, 4-bit, 8-bit quantized)
    # and can automatically download models from HuggingFace Hub.
    
    # Resolve model weights file path
    # Priority order: command line args > config file > auto-selection based on quantization
    model_file = args.moshi_weight
    tokenizer_file = args.tokenizer
    
    # Auto-select appropriate model file if not explicitly provided
    if model_file is None:
        # Check if model name is specified in config
        if type(lm_config) is dict and "moshi_name" in lm_config:
            model_file = hf_hub_download(args.hf_repo, lm_config["moshi_name"])
        # Select quantized model based on command line flag
        elif args.quantized == 8:
            model_file = hf_hub_download(args.hf_repo, "model.q8.safetensors")
        elif args.quantized == 4:
            model_file = hf_hub_download(args.hf_repo, "model.q4.safetensors")
        elif args.quantized is not None:
            # Invalid quantization value provided
            raise ValueError(f"Invalid quantized value: {args.quantized}")
        else:
            # Default to full precision model
            model_file = hf_hub_download(args.hf_repo, "model.safetensors")
    
    # Resolve local path (handles both local files and hf:// URLs)
    model_file = hf_get(model_file)
    
    # Resolve tokenizer file path with similar logic
    if tokenizer_file is None:
        if type(lm_config) is dict and "tokenizer_name" in lm_config:
            tokenizer_file = hf_hub_download(args.hf_repo, lm_config["tokenizer_name"])
        else:
            # Default tokenizer for Moshi models
            tokenizer_file = hf_hub_download(args.hf_repo, "tokenizer_spm_32k_3.model")
    
    # Convert to local path if it's a remote reference
    tokenizer_file = hf_get(tokenizer_file)
    
    # Store generation steps limit from command line args
    steps = args.steps

    # Load the SentencePiece text tokenizer for converting between text and tokens
    # SentencePiece is used for subword tokenization, handling spaces with special tokens
    log("info", f"[SERVER] loading text tokenizer {tokenizer_file}")
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_file)  # type: ignore
    
    # =========================================================================
    # Model Initialization
    # =========================================================================
    # This section initializes the language model with proper configuration,
    # applies quantization for efficiency, and loads the trained weights.
    
    # Set random seed for reproducibility - ensures consistent behavior across runs
    # Using a fixed seed (299792458) for deterministic results
    mx.random.seed(299792458)
    
    # Create model from config - convert dict config to LmConfig object if needed
    # The config defines model architecture parameters like layer count, hidden size, etc.
    if type(lm_config) is dict:
        lm_config = models.LmConfig.from_config_dict(lm_config)
    
    # Instantiate the language model with the specified configuration
    model = models.Lm(lm_config)
    
    # Set model precision to bfloat16 (brain floating point 16-bit)
    # This provides a good balance between precision and memory usage
    model.set_dtype(mx.bfloat16)
    
    # Apply quantization if requested - significantly reduces memory footprint and increases inference speed
    # 4-bit quantization uses group_size=32, 8-bit uses group_size=64
    # Quantization converts weights from floating point to lower precision integers
    if args.quantized is not None:
        group_size = 32 if args.quantized == 4 else 64
        nn.quantize(model, bits=args.quantized, group_size=group_size)

    # Load the pre-trained model weights from the specified file
    # strict=True ensures all weights in the file are loaded and no extra weights are expected
    log("info", f"[SERVER] loading weights {model_file}")
    model.load_weights(model_file, strict=True)
    log("info", "[SERVER] weights loaded")

    # =========================================================================
    # Conditioning Setup
    # =========================================================================
    # Configure model conditioning for quality control and prepare for inference.
    # Conditioning tensors guide the model to produce higher quality outputs.
    
    # Get condition tensor for quality control (if model supports conditioning)
    # The condition tensor provides guidance to the model for generating high-quality outputs
    # "very_good" is a predefined quality level that optimizes for natural speech
    if model.condition_provider is not None:
        ct = model.condition_provider.condition_tensor("description", "very_good")
    else:
        # Model doesn't support conditioning - use None
        ct = None

    # Warmup the model - this is a crucial step that:
    # 1. Compiles MLX kernels for the specific model architecture
    # 2. Allocates memory for model parameters and intermediate computations
    # 3. Initializes internal model state
    # 4. Ensures the model is ready for real-time inference
    log("info", "[SERVER] warming up the model")
    model.warmup(ct)
    log("info", "[SERVER] model warmed up")
    
    # Create the generation wrapper that manages the inference process
    # LmGen handles the autoregressive generation loop and sampling strategies
    gen = models.LmGen(
        model=model,                    # The loaded language model
        max_steps=steps + 5,            # Maximum generation steps (with buffer)
        text_sampler=utils.Sampler(),   # Sampling strategy for text tokens
        audio_sampler=utils.Sampler(),  # Sampling strategy for audio tokens
        check=False,                   # Disable safety checks for performance
    )

    # =========================================================================
    # Main Inference Loop
    # =========================================================================
    # The core real-time inference loop that processes audio tokens and generates
    # text and audio responses. This loop runs continuously until interrupted.
    
    # Signal to the web server that the model server is ready to receive requests
    # This handshake ensures proper synchronization between processes
    server_to_client.put("start")
    log("info", "[SERVER] connected!")
    
    try:
        # Main processing loop - runs indefinitely for real-time conversation
        while True:
            # Receive audio tokens from the client (web server process)
            # These tokens represent the user's spoken input encoded by Mimi
            data = client_to_server.get()
            
            # Reshape tokens from [codebooks, 1] to [1, main_codebooks]
            # This prepares the data for the model's expected input format
            # We only use the main codebooks (first N codebooks) for generation
            data = mx.array(data).transpose(1, 0)[:, : gen.main_codebooks]
            
            # Run one step of autoregressive generation
            # The model processes the input tokens and generates the next text/audio tokens
            # ct (condition tensor) guides the generation for quality
            text_token = gen.step(data, ct=ct)
            text_token = text_token[0].item()  # Extract scalar value
            audio_tokens = gen.last_audio_tokens()  # Get generated audio tokens
            
            # Send text token if it's meaningful (not padding or end-of-sequence)
            # Padding token (0) and EOS token (3) are filtered out
            if text_token not in (0, 3):
                # Convert token ID to text piece using the tokenizer
                _text = text_tokenizer.id_to_piece(text_token)  # type: ignore
                # Replace SentencePiece's special space character (▁) with regular space
                _text = _text.replace("▁", " ")
                # Send text response to client with kind=1
                server_to_client.put_nowait((1, _text))
            
            # Send audio tokens if available (generated speech)
            if audio_tokens is not None:
                # Convert to numpy array with uint32 precision for efficient transfer
                audio_tokens = np.array(audio_tokens).astype(np.uint32)
                # Send audio tokens to client with kind=0
                server_to_client.put_nowait((0, audio_tokens))
    except KeyboardInterrupt:
        # Gracefully exit on Ctrl+C or process termination
        pass


def web_server(client_to_server, server_to_client, lm_config, args):
    """
    Web server process that handles audio I/O and WebSocket connections.
    
    This function runs in a separate process and manages:
    1. Loading and initializing the Mimi audio codec
    2. Running the aiohttp web server with WebSocket support
    3. Encoding/decoding audio between PCM and neural codec tokens
    4. Bridging the WebSocket connection to the model server via queues
    
    The web server handles multiple async tasks concurrently:
    - send_loop: Encodes incoming PCM audio to tokens
    - send_loop2: Forwards encoded tokens to the model server
    - recv_loop: Decodes tokens from the model to PCM audio
    - recv_loop2: Receives tokens from the model server
    
    WebSocket Protocol:
    - Client sends: 0x01 + Opus audio data
    - Server sends: 0x00 (handshake), 0x01 + Opus audio, 0x02 + UTF-8 text
    
    Args:
        client_to_server: Queue for sending audio tokens to the model
        server_to_client: Queue for receiving tokens from the model
        lm_config: Language model configuration
        args: Command-line arguments with server settings
    """
    # =========================================================================
    # Audio Tokenizer Initialization
    # =========================================================================
    
    # Resolve Mimi weights file path
    mimi_file = args.mimi_weight
    if mimi_file is None:
        if type(lm_config) is dict and "mimi_name" in lm_config:
            mimi_file = hf_hub_download(args.hf_repo, lm_config["mimi_name"])
        else:
            mimi_file = hf_hub_download(
                args.hf_repo, "tokenizer-e351c8d8-checkpoint125.safetensors"
            )
    mimi_file = hf_get(mimi_file)
    
    # Create queues for internal audio pipeline
    input_queue = queue.Queue()   # PCM from WebSocket -> encoder
    output_queue = queue.Queue()  # Decoded PCM -> WebSocket
    text_queue = queue.Queue()    # Text tokens -> WebSocket
    
    # Determine number of codebooks from config
    if type(lm_config) is dict:
        nc = lm_config.get("dep_q", 8)
        max_delay = max(lm_config["delays"])
    else:
        nc = lm_config.depformer.num_slices
        max_delay = max(lm_config.audio_delays)
    
    # Initialize the streaming audio tokenizer
    audio_tokenizer = rustymimi.StreamTokenizer(mimi_file, num_codebooks=nc)  # type: ignore
    
    # Wait for the model server to be ready
    start = server_to_client.get()
    log("info", f"[CLIENT] received '{start}' from server, starting...")

    # Warmup the audio pipeline
    full_warmup(audio_tokenizer, client_to_server, server_to_client, max_delay)

    # =========================================================================
    # Async Audio Processing Loops
    # =========================================================================
    
    async def send_loop():
        """
        Encode PCM audio from the input queue to neural codec tokens.
        
        Continuously polls the input queue for PCM frames and feeds them
        to the audio tokenizer's encoder.
        """
        while True:
            await asyncio.sleep(0.001)
            try:
                pcm_data = input_queue.get(block=False)
                audio_tokenizer.encode(pcm_data)
            except queue.Empty:
                continue

    async def recv_loop():
        """
        Retrieve decoded PCM audio and queue it for WebSocket transmission.
        
        Polls the audio tokenizer for decoded PCM frames and adds them
        to the output queue for sending to the client.
        """
        while True:
            data = audio_tokenizer.get_decoded()
            if data is None:
                await asyncio.sleep(0.001)
                continue
            output_queue.put_nowait(data)

    async def send_loop2():
        """
        Forward encoded audio tokens to the model server.
        
        Retrieves encoded tokens from the audio tokenizer and sends them
        to the model server via the inter-process queue.
        """
        while True:
            data = audio_tokenizer.get_encoded()
            if data is None:
                await asyncio.sleep(0.001)
                continue
            client_to_server.put_nowait(data)

    async def recv_loop2():
        """
        Receive tokens from the model server and route them appropriately.
        
        Handles two types of messages from the model:
        - kind=0: Audio tokens -> send to decoder
        - kind=1: Text tokens -> queue for WebSocket
        """
        while True:
            try:
                kind, data = server_to_client.get(block=False)
                if kind == 0:
                    audio_tokenizer.decode(data)
                elif kind == 1:
                    text_queue.put_nowait(data)
            except queue.Empty:
                await asyncio.sleep(0.001)
                continue

    # Lock to ensure only one WebSocket connection at a time
    lock = asyncio.Lock()

    async def handle_chat(request):
        """
        WebSocket handler for real-time audio chat sessions.
        
        Manages a single chat session with bidirectional audio streaming:
        - Receives Opus-encoded audio from the browser
        - Decodes to PCM and feeds to the audio tokenizer
        - Sends model-generated audio and text back to the browser
        
        The handler uses a lock to ensure only one active session at a time,
        as the model maintains state that can't be shared between sessions.
        
        Args:
            request: The aiohttp request object
        
        Returns:
            The WebSocket response object
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        async def recv_loop():
            """
            Receive and process audio from the WebSocket client.
            
            Handles incoming WebSocket messages:
            - Binary messages with kind=1 contain Opus audio
            - Audio is decoded and accumulated into FRAME_SIZE chunks
            - Complete frames are queued for the audio tokenizer
            """
            nonlocal close
            all_pcm_data = None
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        log("error", f"{ws.exception()}")
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        log("error", f"unexpected message type {message.type}")
                        continue
                    message = message.data
                    if not isinstance(message, bytes):
                        log("error", f"unsupported message type {type(message)}")
                        continue
                    if len(message) == 0:
                        log("warning", "empty message")
                        continue
                    
                    # First byte indicates message type
                    kind = message[0]
                    if kind == 1:  # Audio data
                        payload = message[1:]
                        # Decode Opus to PCM
                        pcm = opus_reader.append_bytes(payload)
                        if pcm.shape[-1] == 0:
                            continue
                        
                        # Accumulate PCM samples
                        if all_pcm_data is None:
                            all_pcm_data = pcm
                        else:
                            all_pcm_data = np.concatenate((all_pcm_data, pcm))
                        
                        # Queue complete frames for processing
                        while all_pcm_data.shape[-1] >= FRAME_SIZE:
                            chunk = all_pcm_data[:FRAME_SIZE]
                            all_pcm_data = all_pcm_data[FRAME_SIZE:]
                            input_queue.put_nowait(chunk)

                    else:
                        log("warning", f"unknown message kind {kind}")
            finally:
                close = True
                log("info", "connection closed")

        async def send_loop():
            """
            Send audio and text to the WebSocket client.
            
            Continuously polls the output queues and sends:
            - 0x01 + Opus audio for model-generated speech
            - 0x02 + UTF-8 text for transcribed/generated text
            """
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                try:
                    # Get decoded PCM audio
                    pcm_data = output_queue.get(block=False)
                    assert pcm_data.shape == (1920,), pcm_data.shape
                    
                    # Encode to Opus and send
                    msg = opus_writer.append_pcm(pcm_data)
                    if len(msg) > 0:
                        await ws.send_bytes(b"\x01" + msg)
                    
                    # Send any pending text
                    _text = text_queue.get(block=False)
                    await ws.send_bytes(b"\x02" + bytes(_text, encoding="utf8"))
                except queue.Empty:
                    continue

        log("info", "accepted connection")
        close = False
        
        # Acquire lock to ensure single session
        async with lock:
            log("info", "lock acquired")
            
            # Initialize Opus codec streams
            opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)
            opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)
            
            # Send handshake byte to signal ready
            await ws.send_bytes(b"\x00")
            
            # Run send and receive loops concurrently
            await asyncio.gather(recv_loop(), send_loop())
        
        log("info", "done with connection")
        return ws

    async def go():
        """
        Main async function that sets up and runs the web server.
        
        Configures the aiohttp application with:
        - WebSocket endpoint at /api/chat
        - Static file serving for the web interface
        - Optional SSL/HTTPS support
        
        Also starts all the background audio processing loops.
        """
        # Create the web application
        app = web.Application()
        app.router.add_get("/api/chat", handle_chat)
        
        # =====================================================================
        # Static Content Setup
        # =====================================================================
        
        static_path: None | str = None
        if args.static is None:
            # Download and extract the default web interface
            log("info", "retrieving the static content")
            dist_tgz = hf_hub_download("kyutai/moshi-artifacts", "dist.tgz")
            dist_tgz = Path(dist_tgz)
            dist = dist_tgz.parent / "dist"
            if not dist.exists():
                with tarfile.open(dist_tgz, "r:gz") as tar:
                    tar.extractall(path=dist_tgz.parent)
            static_path = str(dist)
        elif args.static != "none":
            # Use custom static content path
            # When set to "none", no static content is served
            static_path = args.static
        
        if static_path is not None:
            async def handle_root(_):
                """Serve index.html for the root path."""
                return web.FileResponse(os.path.join(static_path, "index.html"))

            log("info", f"serving static content from {static_path}")
            app.router.add_get("/", handle_root)
            app.router.add_static("/", path=static_path, name="static")
        
        # =====================================================================
        # Server Configuration
        # =====================================================================
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        # Configure SSL if requested
        ssl_context = None
        protocol = "http"
        if args.ssl is not None:
            import ssl

            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            cert_file = os.path.join(args.ssl, "cert.pem")
            key_file = os.path.join(args.ssl, "key.pem")
            ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)
            protocol = "https"
        
        # Create and start the TCP site
        site = web.TCPSite(runner, args.host, args.port, ssl_context=ssl_context)

        log("info", f"listening to {protocol}://{args.host}:{args.port}")

        # Optionally open browser
        if not args.no_browser:
            log("info", f"opening browser at {protocol}://{args.host}:{args.port}")
            webbrowser.open(f"{protocol}://{args.host}:{args.port}")

        # Run all async tasks concurrently
        await asyncio.gather(
            recv_loop(), send_loop(), recv_loop2(), send_loop2(), site.start()
        )
        await runner.cleanup()

    # Run the async event loop
    try:
        asyncio.run(go())
    except KeyboardInterrupt:
        pass


def main():
    """
    Main entry point for the web-based Moshi inference server.
    
    Parses command-line arguments, initializes the multi-process architecture,
    and manages the lifecycle of the web server and model server processes.
    
    The function:
    1. Parses CLI arguments for model configuration
    2. Sets up inter-process communication queues
    3. Loads the model configuration
    4. Spawns web server and model server processes
    5. Monitors processes and handles graceful shutdown
    
    Command-line Arguments:
        --tokenizer: Path to the SentencePiece tokenizer file
        --moshi-weight: Path to the Moshi model weights
        --mimi-weight: Path to the Mimi codec weights
        -q/--quantized: Quantization level (4 or 8 bits)
        --steps: Maximum generation steps (default: 4000)
        --hf-repo: HuggingFace repository for model files
        --static: Path to static web content (or "none")
        --host: Server host address (default: localhost)
        --port: Server port (default: 8998)
        --lm-config: Path to LM config JSON file
        --ssl: Directory containing SSL certificates
        --no-browser: Don't auto-open browser
    """
    # =========================================================================
    # Argument Parsing
    # =========================================================================
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--moshi-weight", type=str)
    parser.add_argument("--mimi-weight", type=str)
    parser.add_argument("-q", "--quantized", type=int, choices=[4, 8])
    parser.add_argument("--steps", default=4000, type=int)
    parser.add_argument("--hf-repo", type=str)
    parser.add_argument("--static", type=str)
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--lm-config", type=str, help="The LM config as a json file.")
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        ),
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--debug-model",
        action="store_true",
        help="Enable debugpy for model_server process (attach on port 5678)",
    )

    args = parser.parse_args()
    
    # =========================================================================
    # Default Repository Selection
    # =========================================================================
    
    # Auto-select HuggingFace repo based on quantization setting
    if args.hf_repo is None:
        if args.quantized == 8:
            args.hf_repo = "kyutai/moshiko-mlx-q8"
        elif args.quantized == 4:
            args.hf_repo = "kyutai/moshiko-mlx-q4"
        elif args.quantized is None:
            args.hf_repo = "kyutai/moshiko-mlx-bf16"
        else:
            print(f"Invalid value for quantized {args.quantized}")
            sys.exit(1)

    # =========================================================================
    # Inter-Process Communication Setup
    # =========================================================================
    
    # Create queues for bidirectional communication between processes
    client_to_server = multiprocessing.Queue()  # Audio tokens -> Model
    server_to_client = multiprocessing.Queue()  # Generated tokens -> Web server

    # =========================================================================
    # Model Configuration Loading
    # =========================================================================
    
    lm_config = args.lm_config
    if lm_config is None:
        # Try to download config from HuggingFace
        try:
            lm_config = hf_hub_download(args.hf_repo, "config.json")
        except Exception:
            log("warning", "Cannot download config, using defaults.")
    
    if lm_config is None:
        # Fall back to default v0.1 configuration
        lm_config = models.config_v0_1()
    else:
        # Load config from JSON file
        with open(hf_get(lm_config), "r") as fobj:
            lm_config = json.load(fobj)

    # =========================================================================
    # Process Creation and Management
    # =========================================================================
    
    # Create the two worker processes
    subprocess_args = client_to_server, server_to_client, lm_config, args
    p1 = multiprocessing.Process(target=web_server, args=subprocess_args)
    p2 = multiprocessing.Process(target=model_server, args=subprocess_args)

    # Start both processes
    p1.start()
    p2.start()

    # Monitor processes and handle shutdown
    try:
        while p1.is_alive() and p2.is_alive():
            time.sleep(0.001)
    except KeyboardInterrupt:
        log("warning", "Interrupting, exiting connection.")
        p1.terminate()
        p2.terminate()

    # Wait for processes to complete
    p1.join()
    p2.join()
    log("info", "All done!")


if __name__ == "__main__":
    main()
