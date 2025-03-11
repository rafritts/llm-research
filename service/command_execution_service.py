from typing import Dict, Any, Optional, Tuple, Union
import json
from tools.base import format_tool_specifications
from tools.command_execution import CommandExecutionTool
from tools.rag_search import RAGSearchTool
from tools.chain_of_thought import ChainOfThoughtTool
from models.schema import ToolResponse
from llm.ollama import generate_response_stream


async def execute_tool(tool_name: str, parameters: Dict[str, Any]) -> Optional[ToolResponse]:
    """Execute the specified tool with given parameters"""
    tool_map = {
        "command_execution": CommandExecutionTool,
        "rag_search": RAGSearchTool,
        "chain_of_thought": ChainOfThoughtTool
    }

    tool_class = tool_map.get(tool_name)
    if not tool_class:
        return None

    return await tool_class.execute(parameters)


async def parse_and_execute_tool_call(response: str) -> Optional[ToolResponse]:
    """Parse a potential tool call from the LLM response and execute if valid"""
    try:
        # Try to parse the response as JSON
        parsed = json.loads(response)
        
        # Check if this is a tool call
        if isinstance(parsed, dict) and "tool_calls" in parsed:
            tool_calls = parsed["tool_calls"]
            if tool_calls and len(tool_calls) > 0:
                tool_call = tool_calls[0]  # We only support one tool call at a time
                return await execute_tool(tool_call["tool"], tool_call["parameters"])
                
    except json.JSONDecodeError:
        # Not a JSON response, so not a tool call
        pass
    except Exception as e:
        # Log any other errors but don't raise
        print(f"Error executing tool: {str(e)}")
        
    return None


async def generate_tool_execution_prompt(
    query: str,
    model: str,
    temperature: float
) -> str:
    """Generate a prompt for tool execution, execute any tool calls, and return final response"""
    tool_specs = format_tool_specifications()
    initial_prompt = f"""{tool_specs}
                {query}"""
    
    tool_prompt = f"""When you need to use a tool, respond with JSON in the following format:
            {{
                "thoughts": "your reasoning about what tool to use and why",
                "tool_calls": [
                    {{
                        "tool": "tool_name",
                        "parameters": {{
                            "param1": "value1"
                        }}
                    }}
                ]
            }}
            You can only use one tool. If you don't need to use a tool, just respond normally.

            {initial_prompt}"""

    # Get initial response
    buffer = ""
    first_chunk = True
    async for chunk in generate_response_stream(tool_prompt, model=model, temperature=temperature):
        if first_chunk and "{" in chunk:
            buffer = chunk
            async for next_chunk in generate_response_stream(tool_prompt, model=model, temperature=temperature):
                buffer += next_chunk
                if "}" in buffer:
                    break
            
            # Try to execute tool call if present
            result = await parse_and_execute_tool_call(buffer)
            if result:
                # Generate followup response based on tool execution
                followup_prompt = f"""Based on the tool execution results:
                                    {result.output or 'No output'}
                                    {f"Error: {result.error}" if result.error else ""}
                                    Please provide a final response to: {query}"""
                final_response = ""
                async for chunk in generate_response_stream(followup_prompt, model=model, temperature=temperature):
                    final_response += chunk
                return final_response
            
        first_chunk = False
        buffer += chunk

    return buffer  # Return accumulated response if no tool was executed