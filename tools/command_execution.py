import logging
import subprocess
import shlex
from typing import Dict, Any
from models.schema import ToolResponse
from config import ALLOWED_COMMANDS
from tools.base import BaseTool

logger = logging.getLogger(__name__)

class CommandExecutionTool(BaseTool):
    name = "command_execution"
    description = "Execute a shell command and return its output"
    
    @classmethod
    async def execute(cls, parameters: Dict[str, Any]):
        """Execute a shell command with safety checks"""
        command = parameters.get("cli_command", "")
        logger.info(f"Attempting to execute command: {command}")
        
        # Parse the command to get the base command
        try:
            cmd_parts = shlex.split(command)
            base_cmd = cmd_parts[0]
            logger.debug(f"Parsed command parts: {cmd_parts}")
        except Exception as e:
            logger.error(f"Command parsing failed: {str(e)}")
            return ToolResponse(
                output="", error=f"Command parsing error: {str(e)}", exit_code=1
            )

        # Security check - only allow specific commands
        if base_cmd not in ALLOWED_COMMANDS:
            logger.warning(f"Blocked unauthorized command: {base_cmd}")
            return ToolResponse(
                output="",
                error=f"Command '{base_cmd}' is not allowed for security reasons. Allowed commands: {', '.join(ALLOWED_COMMANDS)}",
                exit_code=1,
            )

        logger.info(f"Executing validated command: {command}")
        # Execute the command
        try:
            # Set a timeout to prevent long-running commands
            process = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,  # 10-second timeout
            )
            
            if process.returncode == 0:
                logger.info("Command executed successfully")
                logger.debug(f"Command output: {process.stdout}")
            else:
                logger.warning(f"Command failed with exit code {process.returncode}")
                logger.debug(f"Command error output: {process.stderr}")
                
            return ToolResponse(
                output=process.stdout,
                error=process.stderr if process.stderr else None,
                exit_code=process.returncode,
            )
        except subprocess.TimeoutExpired:
            logger.error("Command execution timed out")
            return ToolResponse(
                output="",
                error="Command execution timed out after 10 seconds",
                exit_code=124,
            )
        except Exception as e:
            logger.exception("Command execution failed with unexpected error")
            return ToolResponse(
                output="", error=f"Command execution error: {str(e)}", exit_code=1
            )