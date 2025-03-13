from typing import Dict, Any, Optional
import json
import logging
from models.schema import ToolResponse
from tools.command_execution import CommandExecutionTool

# Configure logging
logger = logging.getLogger(__name__)

async def execute_command_tool(cli_request: Dict[str, Any]) -> Optional[ToolResponse]:
    """Execute a command line tool with the given parameters"""
    try:
        logger.info(f"Received tool call: {json.dumps(cli_request)}")
        
        # Create parameters dictionary
        parameters = {"cli_command": cli_request["cli_command"]}
        
        logger.info(f"Executing command: {json.dumps(parameters['cli_command'])}")
        result = await CommandExecutionTool.execute(parameters)
        
        if result.error:
            logger.error(f"Command execution failed: {result.error}")
        else:
            logger.info("Command executed successfully")
            logger.debug(f"Command output: {result.output}")
            
        return result
                
    except Exception as e:
        logger.exception("Error executing command")
        return ToolResponse(output="", error=f"Error executing command: {str(e)}")