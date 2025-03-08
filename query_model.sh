#!/bin/bash
# Check if query parameter is provided
if [ -z "$1" ]; then
  echo "Usage: ./query_model.sh 'Your question here' [context_window] [temperature] [use_cot]"
  echo "  use_cot: Set to 'true' to enable Chain of Thought processing"
  exit 1
fi

# Optional parameters with defaults
CONTEXT_WINDOW=${2:-5}   # Default to 5 documents
TEMPERATURE=${3:-0.7}    # Default temperature 0.7
USE_COT=${4:-false}      # Default Chain of Thought to false

# Create request body and send it directly to curl
echo "Sending query: $1"
echo "Chain of Thought: $USE_COT"

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  --data-raw "{
    \"query\": \"$1\", 
    \"context_window\": $CONTEXT_WINDOW, 
    \"temperature\": $TEMPERATURE,
    \"use_chain_of_thought\": $USE_COT
  }" \
  --max-time 300

echo -e "\n"