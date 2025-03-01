#!/bin/bash

# Function to store a document
store_document() {
    local filename="$1"
    local EMBEDDING
    local JSON_RESPONSE
    
    echo "Processing $filename..."
    
    # Get embeddings from Ollama
    JSON_RESPONSE=$(curl -s http://localhost:11434/api/embeddings \
        -d "$(jq -n --arg prompt "$(cat "$filename")" '{model: "gemma2:27b", prompt: $prompt}')")
    
    # Check if the response is valid JSON
    if [[ "$JSON_RESPONSE" == *'"embedding":'* ]]; then
        # Extract the embedding from the JSON response
        EMBEDDING=$(echo "$JSON_RESPONSE" | jq -c '.embedding')
        
        # Check if jq returned anything
        if [ -z "$EMBEDDING" ]; then
            echo "Error: Failed to extract embedding from Ollama response."
            echo "Ollama response: $JSON_RESPONSE"
            return 1 # Indicate an error
        fi
        
        # Read file content and escape it for JSON
        CONTENT=$(jq -Rs . < "$filename")
        
        # Create a clean temporary file with the JSON payload
        TEMP_FILE=$(mktemp)
        cat > "$TEMP_FILE" << EOF
{
  "content": $CONTENT,
  "embedding": $EMBEDDING,
  "metadata": {"filename": "$(basename "$filename")"}
}
EOF
        
        echo "Sending request to store document..."
        
        # Use the temporary file for the request to avoid shell escaping issues
        RESPONSE=$(curl -s -X POST http://localhost:8000/embed \
            -H "Content-Type: application/json" \
            --data @"$TEMP_FILE")
        
        echo "Response from server:"
        echo "$RESPONSE"
        
        # Clean up
        rm "$TEMP_FILE"
        
        if [[ "$RESPONSE" == *"success"* ]]; then
            echo "Document '$filename' successfully stored in the database!"
        else
            echo "WARNING: Document storage may have failed."
        fi
    else
        echo "Error: Ollama returned an error response. Skipping document."
        return 1
    fi
}

# Process each document
#for doc in doc1.txt doc2.txt doc3.txt doc4.txt; do
for doc in doc1.txt; do
    if [ -f "$doc" ]; then
        store_document "$doc"
        echo "--------------------------------------------"
    else
        echo "$doc not found"
    fi
done

echo "All documents processed."