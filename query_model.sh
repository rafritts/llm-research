#!/bin/bash
# Optional parameters with defaults
CONTEXT_WINDOW=${2:-5} # Default to 5 documents
TEMPERATURE=${3:-0.7} # Default temperature 0.7
query="Describe the steps for how to build a house by hand. Assume its in the woods, and its a small cabin."

# Function to show a spinner
spinner() {
  local delay=0.1
  local spinstr='|/-\'
  echo -n "Connecting to API "
  for i in {1..20}; do  # Show spinner for ~2 seconds max while connecting
    local temp=${spinstr#?}
    printf "[%c]" "$spinstr"
    local spinstr=$temp${spinstr%"$temp"}
    sleep $delay
    printf "\b\b\b"
  done
  printf "    \b\b\b\b"
  echo -e "\nResponse:\n"
}

# Show the spinner first
spinner

# Now execute the curl command with streaming output
curl -N -X POST http://localhost:8000/query \
-H "Content-Type: application/json" \
-H "Accept: text/event-stream" \
--data-raw "{\"query\": \"$query\", \"context_window\": $CONTEXT_WINDOW, \"temperature\": $TEMPERATURE, \"use_cot\": true, \"use_rag\": false, \"allow_tools\": true}"

echo -e "\n"