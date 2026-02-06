#!/bin/bash

# Moshi MLX Setup Script for macOS
# This script creates a virtual environment and installs moshi_mlx

set -e  # Exit on any error

echo "🚀 Setting up Moshi MLX for macOS..."

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is designed for macOS only"
    exit 1
fi

# Check for pyenv and suggest it for Python version management
if command -v pyenv >/dev/null 2>&1; then
    echo "✅ pyenv detected - good for Python version management"
else
    echo "💡 Consider installing pyenv for better Python version management:"
    echo "   brew install pyenv"
    echo "   pyenv install 3.12.0"
    echo "   pyenv local 3.12.0"
    echo ""
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.10"

# Note: Python 3.14 may need NO_TORCH_COMPILE=1 for some packages
if [[ "$PYTHON_VERSION" == "3.14" ]]; then
    echo "⚠️  Python 3.14 is very new - some packages may not be compatible"
    echo "💡 Recommended: Use Python 3.12 for better compatibility"
    echo "   Option 1: brew install python@3.12 && python3.12 -m venv moshi_env"
    echo "   Option 2: pyenv install 3.12.0 && pyenv local 3.12.0"
    echo "🔧 Setting compatibility flags for Python 3.14..."
    export NO_TORCH_COMPILE=1
    export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
fi

if [[ $(echo "$PYTHON_VERSION >= $REQUIRED_VERSION" | bc -l) -eq 0 ]]; then
    echo "❌ Python $REQUIRED_VERSION+ required, found $PYTHON_VERSION"
    echo "💡 Install newer Python with: brew install python@3.12"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Check for required build tools
echo "🔧 Checking build dependencies..."
MISSING_TOOLS=()

if ! command -v pkg-config >/dev/null 2>&1; then
    MISSING_TOOLS+=("pkg-config")
fi

if ! command -v cmake >/dev/null 2>&1; then
    MISSING_TOOLS+=("cmake")
fi

if ! command -v nproc >/dev/null 2>&1 && ! command -v gnproc >/dev/null 2>&1; then
    MISSING_TOOLS+=("coreutils")
fi

if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
    echo "⚠️  Missing build tools: ${MISSING_TOOLS[*]}"
    echo "💡 Installing with Homebrew..."
    brew install "${MISSING_TOOLS[@]}"
fi

# Add GNU coreutils to PATH if needed
if command -v gnproc >/dev/null 2>&1; then
    export PATH="/opt/homebrew/opt/coreutils/libexec/gnubin:$PATH"
fi

echo "✅ Build dependencies ready"

# Create virtual environment if it doesn't exist
VENV_DIR="moshi_env"
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv $VENV_DIR
else
    echo "📦 Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source $VENV_DIR/bin/activate

# Check pip version and inform user if upgrade needed
echo "🔧 Checking pip version..."
PIP_VERSION=$(pip --version | grep -o '[0-9]\+\.[0-9]\+' | head -1)
REQUIRED_PIP="21.0"

if [[ $(echo "$PIP_VERSION >= $REQUIRED_PIP" | bc -l) -eq 0 ]]; then
    echo "⚠️  Your pip version ($PIP_VERSION) is quite old."
    echo "💡 If installation fails, please run: pip install --upgrade pip"
    echo "   Then re-run this script."
else
    echo "✅ pip $PIP_VERSION looks good"
fi

# Check if we're in the moshi repository
if [ -d "moshi_mlx" ] && [ -f "moshi_mlx/pyproject.toml" ]; then
    echo "🏠 Installing local moshi_mlx in development mode..."
    pip install -e 'moshi_mlx[dev]'
    INSTALL_TYPE="local development"
else
    echo "📥 Installing moshi_mlx from PyPI..."
    pip install -U moshi_mlx
    INSTALL_TYPE="PyPI"
fi

# Install debugpy for debugging support
echo "🐛 Installing debugpy for debugging support..."
pip install debugpy

# Verify installation
echo "🧪 Verifying installation..."
python -c "import moshi_mlx; print('✅ moshi_mlx installed successfully')"

echo ""
echo "🎉 Setup complete!"
echo "📋 Installation summary:"
echo "   • Python: $PYTHON_VERSION"
echo "   • Virtual environment: $VENV_DIR"
echo "   • moshi_mlx source: $INSTALL_TYPE"
echo ""
echo "🚀 To use Moshi MLX:"
echo "   1. Activate environment: source $VENV_DIR/bin/activate"
echo "   2. Run with 4-bit quantization: moshi-local -q 4"
echo "   3. Or run web UI: moshi-local-web -q 4"
echo "   4. Deactivate when done: deactivate"
echo ""
echo "💡 Daily workflow:"
echo "   source moshi_env/bin/activate && moshi-local -q 4"
echo ""
echo "💡 Pro tip: The first run will download ~2-4GB of model weights"