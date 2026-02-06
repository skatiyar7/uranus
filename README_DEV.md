# Moshi Development Setup

Quick development setup guide for Moshi - a real-time speech-to-speech AI conversation system.

## 🚀 Quick Start (macOS)

### One-Time Setup
```bash
# Clone the repository (if not already done)
git clone https://github.com/kyutai-labs/moshi.git
cd moshi

# Run the initialization script
./init_mac_dev.sh
```

### Daily Usage
```bash
# Activate environment and run Moshi
source moshi_env/bin/activate && moshi-local -q 4

# Or step by step
source moshi_env/bin/activate
moshi-local -q 4                      # CLI interface
moshi-local-web -q 4                  # Web UI at http://localhost:8998
deactivate                            # When done
```

## 📋 What You Get

- **Virtual Environment**: Isolated Python environment in `moshi_env/`
- **Development Install**: Editable installation of moshi_mlx
- **All Dependencies**: MLX, numpy, safetensors, rustymimi, etc.
- **Dev Tools**: pyright, flake8, pre-commit hooks

## 🛠️ Development Workflow

### Making Changes
```bash
# Activate environment
source moshi_env/bin/activate

# Edit code in moshi_mlx/moshi_mlx/
# Changes are immediately available (no reinstall needed)

# Test your changes
moshi-local -q 4

# Run code quality checks
flake8 moshi_mlx/
pyright moshi_mlx/
```

### Model Options
```bash
# Different quantization levels
moshi-local -q 4                      # 4-bit (most memory efficient)
moshi-local -q 8                      # 8-bit (balanced)
moshi-local                           # bf16 (full precision)

# Different voices
moshi-local -q 4 --hf-repo kyutai/moshika-mlx-q4  # Female
moshi-local -q 4 --hf-repo kyutai/moshiko-mlx-q4  # Male

# Web interface
moshi-local-web -q 4                  # Web UI at http://localhost:8998
```

## 📁 Project Structure

```
moshi/
├── init_mac_dev.sh           # Setup script
├── moshi_env/               # Virtual environment (created by script)
├── moshi_mlx/               # MLX implementation
│   ├── moshi_mlx/
│   │   ├── models/          # Model implementations
│   │   ├── utils/           # Utilities
│   │   ├── local.py         # CLI interface
│   │   └── local_web.py     # Web interface
│   ├── pyproject.toml       # Package configuration
│   └── requirements.txt     # Dependencies
├── moshi/                   # PyTorch implementation
├── rust/                    # Rust implementation
└── client/                  # Web UI client
```

## 🔧 Requirements

- **macOS**: Apple Silicon recommended
- **Python**: 3.10+ (3.12 recommended)
- **Memory**: 16GB+ RAM
- **Storage**: ~4GB for models (downloaded on first run)

## 🐛 Debugging

Both `moshi-local` and `moshi-local-web` support debugging the model server process via debugpy.

### Prerequisites
```bash
pip install debugpy
```

### Start with Debug Mode
```bash
# CLI interface
moshi-local -q 4 --debug-model

# Web interface
moshi-local-web -q 4 --debug-model
```

The server will pause and display:
```
[Info] [SERVER] Debugger listening on port 5678. Waiting for client...
```

### Attach Debugger (VS Code / Kiro)

1. Create `.vscode/launch.json`:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Attach to model_server",
      "type": "debugpy",
      "request": "attach",
      "connect": {
        "host": "localhost",
        "port": 5678
      },
      "justMyCode": false
    }
  ]
}
```

2. Open Run and Debug panel (⇧⌘D)
3. Select "Attach to model_server" and click play (F5)

Once attached, you'll see:
```
[Info] [SERVER] Debugger attached!
```

Set breakpoints in `moshi_mlx/moshi_mlx/local_web.py` or `local.py` (in the `model_server`/`server` functions) and they'll be hit during execution.

## 🔧 Troubleshooting

### Common Issues

**"pip: command not found"**
```bash
# Use pip3 instead, or install newer Python
brew install python@3.12
```

**"externally-managed-environment"**
```bash
# The init script handles this with virtual environments
./init_mac_dev.sh
```

**"MLX not found"**
```bash
# Ensure you're on Apple Silicon Mac
system_profiler SPHardwareDataType | grep "Chip"
```

**Model download fails**
```bash
# Check internet connection and try again
# Models are ~2-4GB and cached in ~/.cache/huggingface/
```

**Command not found: moshi-local**
```bash
# Make sure virtual environment is activated
source moshi_env/bin/activate
# Or use full path
./moshi_env/bin/moshi-local -q 4
```

### Reset Environment
```bash
# Remove and recreate
rm -rf moshi_env
./init_mac_dev.sh
```

## 🎯 Performance Tips

- **Use 4-bit quantization** (`-q 4`) for best memory efficiency
- **Web UI recommended** - provides echo cancellation
- **First run downloads models** - be patient (~2-4GB)
- **Close other apps** to free up memory during conversation

## 📚 Additional Resources

- [Main README](README.md) - Full project documentation
- [MLX README](moshi_mlx/README.md) - MLX-specific details
- [Moshi Paper](https://arxiv.org/abs/2410.00037) - Technical details
- [Live Demo](https://moshi.chat) - Try it online

## 🤝 Contributing

1. Make your changes in `moshi_mlx/`
2. Test with `moshi-local -q 4`
3. Run quality checks: `flake8` and `pyright`
4. Submit a pull request

## 📄 License

- Code: MIT License (Python), Apache License (Rust)
- Models: CC-BY 4.0 License

---

**Happy coding with Moshi! 🎤🤖**