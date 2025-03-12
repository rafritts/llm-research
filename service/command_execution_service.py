from typing import Dict, Any, Optional
import json
import logging
from models.schema import ToolResponse
from tools.command_execution import CommandExecutionTool

# Configure logging
logger = logging.getLogger(__name__)

async def execute_command_tool(tool_call: Dict[str, Any]) -> Optional[ToolResponse]:
    """Execute a command line tool with the given parameters"""
    try:
        logger.info(f"Received tool call: {json.dumps(tool_call)}")
        
        if not isinstance(tool_call, dict) or "tool" not in tool_call or "parameters" not in tool_call:
            logger.error("Invalid tool call format")
            return ToolResponse(error="Invalid tool call format")
            
        if tool_call["tool"] != "command_execution":
            logger.error(f"Not a command execution tool. Got: {tool_call['tool']}")
            return ToolResponse(error="Not a command execution tool")
        
        logger.info(f"Executing command with parameters: {json.dumps(tool_call['parameters'])}")
        result = await CommandExecutionTool.execute(tool_call["parameters"])
        
        if result.error:
            logger.error(f"Command execution failed: {result.error}")
        else:
            logger.info("Command executed successfully")
            logger.debug(f"Command output: {result.output}")
            
        return result
                
    except Exception as e:
        logger.exception("Error executing command")
        return ToolResponse(error=f"Error executing command: {str(e)}")