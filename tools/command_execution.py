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
        command = parameters.get("command", "")
        
        # Parse the command to get the base command
        try:
            cmd_parts = shlex.split(command)
            base_cmd = cmd_parts[0]
        except Exception as e:
            return ToolResponse(
                output="", error=f"Command parsing error: {str(e)}", exit_code=1
            )

        # Security check - only allow specific commands
        if base_cmd not in ALLOWED_COMMANDS:
            return ToolResponse(
                output="",
                error=f"Command '{base_cmd}' is not allowed for security reasons. Allowed commands: {', '.join(ALLOWED_COMMANDS)}",
                exit_code=1,
            )

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

            return ToolResponse(
                output=process.stdout,
                error=process.stderr if process.stderr else None,
                exit_code=process.returncode,
            )
        except subprocess.TimeoutExpired:
            return ToolResponse(
                output="",
                error="Command execution timed out after 10 seconds",
                exit_code=124,
            )
        except Exception as e:
            return ToolResponse(
                output="", error=f"Command execution error: {str(e)}", exit_code=1
            )