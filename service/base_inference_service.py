from typing import Dict, Any, Optional, AsyncGenerator
import json
import logging
from llm import ollama
from models.schema import ToolResponse
from llm.ollama import generate_response_stream
from config import ALLOWED_COMMANDS
from service.command_execution_service import execute_command_tool

# Configure logging
logger = logging.getLogger(__name__)

async def parse_and_execute_tool_call(response: str) -> Optional[ToolResponse]:
    """Parse a potential tool call from the LLM response and execute if valid"""
    try:
        logger.info("Attempting to parse tool call from response")
        # Clean up markdown and whitespace
        response = response.strip()
        
        if "```" in response:
            # Extract content between code block markers, handling multiple backticks
            parts = response.split("```")
            if len(parts) >= 3:  # Has opening and closing markers
                response = parts[1]
                # Remove any language identifier (e.g. 'json')
                if response.startswith("json"):
                    response = response[4:].strip()
        
        # If response looks like a Python dict string, convert it to JSON format
        if response.startswith("{") and ("'" in response or "}" in response):
            try:
                # Handle Python dict string format
                parsed_dict = eval(response)  # Safe since we know it's a dict string
                response = json.dumps(parsed_dict)  # Convert to proper JSON
            except:
                pass
                
        # Handle escaped quotes in JSON
        response = response.replace('\\"', '"')  # Handle escaped double quotes
        response = response.replace('"%', '%')   # Fix common date format issue
        
        parsed = json.loads(response)
        if isinstance(parsed, dict):
            logger.info(f"Found valid tool call: {json.dumps(parsed)}")
            return await execute_command_tool(parsed)
        else:
            logger.debug("Invalid tool call format - expected a dictionary")
                
    except json.JSONDecodeError as e:
        logger.warning(f"Response is not in JSON format: {str(e)}")
        logger.debug(f"Attempted to parse: {response}")
    except Exception as e:
        logger.exception("Error executing tool")
        print(f"Error executing tool: {str(e)}")
        
    return None

async def generate_response(
    query: str,
    model: str,
    temperature: float
) -> AsyncGenerator[str, None]:
    """Generate a prompt for command execution and handle the conversation flow"""
    logger.info(f"Generating response for query: {query}")
    initial_prompt = f"""You are an AI assistant. 
                     You are able to execute the following command line tools: {ALLOWED_COMMANDS}
    
    Examine the user's request. If the user's request would benefit from executing a command line tool, respond with JSON in this format:
    {{
        "thoughts": "Your thoughts on the how the command will help you answer the user's question",
        "cli_command": "the command to execute"
    }}
    
    For example, to run a date command:
    {{
        "thoughts": "Getting the date",
        "cli_command": "date +%A"
    }}
    
    If no command needs to be run, simply respond to the users prompt.
    
    User prompt: {query}"""

    # Do ollama call
    should_use_command_line_json = ollama.generate_response(initial_prompt, model=model, temperature=temperature)
    logger.info(f"Ollama response: {should_use_command_line_json}")
    # if should_use_command_line_json is a json object, then we should execute the command
    if "{" in should_use_command_line_json or "```json" in should_use_command_line_json.startswith:
        # Execute the command
        logger.info("Should execute command")
        result = await parse_and_execute_tool_call(should_use_command_line_json)
        if result:
            # Stream the command execution results
            tool_result = f"\n\nCommand Execution Results:\n{result.output or 'No output'}"
            if result.error:
                tool_result += f"\nError: {result.error}"
            #yield tool_result

            # Generate and stream followup response
            followup_prompt = f"""Based on the command execution results, that you the AI assistant executed, the output was:
                                {result.output or 'No output'}
                                {f"Error: {result.error}" if result.error else ""}
                                Please provide a final response to: {query}
                                Do not provide a disclaimer for the command execution. The user is aware that the AI assistant executed the command.
                                If the command has failed, offer an apology and tell the user you are still learning how to use the cli.
                                """ 
            
            #yield "\n\nFinal Response:\n"
            async for final_chunk in generate_response_stream(followup_prompt, model=model, temperature=temperature):
                yield final_chunk
            return
    else:
        # Stream the initial prompt
        logger.info("Should not execute command")
        async for chunk in generate_response_stream(query, model=model, temperature=temperature):
            yield chunk

    logger.debug("Response generation completed without tool execution")