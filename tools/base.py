import json
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseTool(ABC):
    name: str
    description: str
    
    @classmethod
    @abstractmethod
    async def execute(cls, parameters: Dict[str, Any]):
        """Execute the tool with the given parameters"""
        pass
    
    @classmethod
    def format_specifications(cls):
        """Format tool specifications for the LLM prompt"""
        return {
            "description": cls.description,
            "parameters": {}  # To be implemented by child classes
        }


def format_tool_specifications():
    """Format all tool specifications for the LLM prompt"""
    from tools.command_execution import CommandExecutionTool
    from tools.rag_search import RAGSearchTool
    from tools.chain_of_thought import ChainOfThoughtTool
    from config import ALLOWED_COMMANDS
    
    tools = {
        "command_execution": {
            "description": CommandExecutionTool.description,
            "parameters": {"command": "The command to execute"},
            "allowed_commands": ALLOWED_COMMANDS,
        },
        "rag_search": {
            "description": RAGSearchTool.description,
            "parameters": {
                "query": "The search query to find relevant information",
                "max_results": "Maximum number of results to return (default: 5)",
            },
        },
        "chain_of_thought": {
            "description": ChainOfThoughtTool.description,
            "parameters": {"question": "The question or problem to think through"},
        },
    }
    return json.dumps(tools, indent=2)