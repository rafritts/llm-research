from typing import List, Dict, Optional, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime

# Tool schemas
class CommandExecutionTool(BaseModel):
    name: str = "command_execution"
    description: str = "Execute a shell command and return its output"

    class Parameters(BaseModel):
        command: str = Field(..., description="The command to execute")

    parameters: Parameters


class RAGSearchTool(BaseModel):
    name: str = "rag_search"
    description: str = "Search for relevant information in the knowledge base"

    class Parameters(BaseModel):
        query: str = Field(
            ..., description="The search query to find relevant information"
        )
        max_results: int = Field(5, description="Maximum number of results to return")

    parameters: Parameters


class ChainOfThoughtTool(BaseModel):
    name: str = "chain_of_thought"
    description: str = "Think step-by-step about complex problems before answering"

    class Parameters(BaseModel):
        question: str = Field(
            ..., description="The question or problem to think through"
        )

    parameters: Parameters


class ToolCall(BaseModel):
    tool: str
    parameters: Dict[str, Any]


class ToolResponse(BaseModel):
    output: str
    error: Optional[str] = None
    exit_code: Optional[int] = None


# Input models
class QueryInput(BaseModel):
    query: str
    temperature: float = 0.7
    allow_tools: bool = True
    model: Optional[str] = None  # Allow specifying a model per request
    use_rag: bool = False  # Flag for automatic RAG processing
    use_cot: bool = False  # Flag for automatic CoT processing


class ModelConfigInput(BaseModel):
    default_model: str
    tool_models: Dict[str, str] = {}


class Document(BaseModel):
    content: str
    embedding: List[float]
    metadata: Dict = {}