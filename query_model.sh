#!/bin/bash
# Check if query parameter is provided
if [ -z "$1" ]; then
  echo "Usage: ./query_model.sh 'Your question here' [context_window] [temperature]"
  exit 1
fi

# Optional parameters with defaults
CONTEXT_WINDOW=${2:-5}   # Default to 5 documents
TEMPERATURE=${3:-0.7}    # Default temperature 0.7

# Create request body and send it directly to curl
echo "Sending query: $1"
curl -N -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  --data-raw "{\"query\": \"$1\", \"context_window\": $CONTEXT_WINDOW, \"temperature\": $TEMPERATURE, \"use_cot\": false, \"use_rag\": false, \"allow_tools\": true}" \
  --max-time 300

echo -e "\n"