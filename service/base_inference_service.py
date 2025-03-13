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
        if response.startswith("```json"):
            response = response.replace("```json", "").replace("```", "")
        parsed = json.loads(response)
        if isinstance(parsed, dict) and "tool_calls" in parsed:
            tool_calls = parsed["tool_calls"]
            if tool_calls and len(tool_calls) > 0:
                logger.info(f"Found valid tool call: {json.dumps(tool_calls[0])}")
                tool_call = tool_calls[0]
                return await execute_command_tool(tool_call)
        else:
            logger.debug("No tool calls found in response")
                
    except json.JSONDecodeError:
        logger.warning("Response is not in JSON format")
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
    
    Examine the user's request. If the user's request would benefit from executing a command line tool, respond with JSON in this format, on a single line:
    {{
        "thoughts": "your reasoning about what command to run and why",
        "tool_calls": [
            {{
                "tool": "command_execution",
                "parameters": {{
                    "command": "the command to execute"
                }}
            }}
        ]
    }}
    If no command needs to be run, simply respond to the users prompt.
    
    User prompt: {query}"""


    # Do ollama call
    should_use_command_line_json = ollama.generate_response(initial_prompt, model=model, temperature=temperature)
    logger.info(f"Ollama response: {should_use_command_line_json}")
    # if should_use_command_line_json is a json object, then we should execute the command
    if should_use_command_line_json.startswith("{") or should_use_command_line_json.startswith("```json"):
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
            followup_prompt = f"""Based on your command execution results:
                                {result.output or 'No output'}
                                {f"Error: {result.error}" if result.error else ""}
                                Please provide a final response to: {query}
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