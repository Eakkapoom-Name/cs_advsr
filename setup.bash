#!/bin/bash

# Setup

echo "⌛️ Setting up in progress..."
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found"

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip first."
    exit 1
fi

echo "✅ pip found"

# Check if virtual environment is created
echo ""
echo "📦 Checking virtual environment..."

VENV_DIR=".venv"
ACTIVATE_PATH="$VENV_DIR/bin/activate"

# Create .venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "📝 Creating .venv"
    python3 -m venv "$VENV_DIR" || { echo "❌ Failed to create .venv"; exit 1; }
    echo "✅ Virtual environment created successfully"
else
    echo "✅ Virtual environment already exists"
fi

# Figure out the absolute path to the expected venv
EXPECTED_VENV_PATH="$(cd "$VENV_DIR" && pwd 2>/dev/null)"

# If the correct venv is already active, do nothing; otherwise, activate it
if [ -n "$VIRTUAL_ENV" ] && [ "$VIRTUAL_ENV" = "$EXPECTED_VENV_PATH" ]; then
    pass
else
    if [ -f "$ACTIVATE_PATH" ]; then
        # shellcheck disable=SC1091
        source "$ACTIVATE_PATH"
    else
        echo "❌ Virtual enviornment activation script not found"
        exit 1
    fi
fi

# Install requirements
echo ""
echo "⚙️ Installing requirements..."

pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Requirements installed successfully"
else
    echo "❌ Failed to install requirements"
    exit 1
fi

echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created. Please edit it with your API keys."
else
    echo "✅ .env file already exists"
fi

echo ""
echo "🎉 Setup complete!"
echo "📝 You can continue to the next step in README.md"
