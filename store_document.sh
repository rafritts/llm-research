#!/bin/bash

# Check if content parameter is provided
if [ -z "$1" ]; then
    echo "Usage: ./store_document.sh 'Your document content'"
    exit 1
fi

# Get embeddings from Ollama first
EMBEDDING=$(curl -s http://localhost:11434/api/embeddings \
    -d "{
        \"model\": \"gemma2:27b\",
        \"prompt\": \"$1\"
    }" | jq -r '.embedding')

# Store document with embedding in Supabase
curl -X POST http://localhost:8000/embed \
    -H "Content-Type: application/json" \
    -d "{
        \"content\": \"$1\",
        \"embedding\": $EMBEDDING,
        \"metadata\": {}
    }"

echo -e "\nDocument stored successfully!"