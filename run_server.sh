#!/bin/bash

# Source the environment variables
if [ -f .env ]; then
    source .env
else
    echo "Error: .env file not found. Please run setup_supabase.sh first."
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/version > /dev/null; then
    echo "Error: Ollama is not running. Please start Ollama first."
    exit 1
fi

# Run the FastAPI server
python serve_llm.py