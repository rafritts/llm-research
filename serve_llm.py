import os
import traceback
import sys
import subprocess
import json
import shlex
from typing import List, Dict, Optional, Union, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from supabase import create_client, Client
import numpy as np
from ollama import Client as OllamaClient
import uvicorn
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Gemma 2 27B with Tool-Based RAG, Command Execution, and Chain-of-Thought",
    root_path_in_servers=False,
    max_request_body_size=1024 * 1024  # 1MB
)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    logger.warning(f"Failed to initialize Supabase client: {str(e)}")
    supabase = None

# Initialize Ollama client
ollama_client = OllamaClient(host='http://localhost:11434')

# Available tools configuration
ALLOWED_COMMANDS = [
    "ls", "pwd", "cat", "grep", "find", "echo", "date", 
    "wc", "head", "tail", "df", "du", "ps"
]

# Define tool schemas
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
        query: str = Field(..., description="The search query to find relevant information")
        max_results: int = Field(5, description="Maximum number of results to return")
    
    parameters: Parameters

class ChainOfThoughtTool(BaseModel):
    name: str = "chain_of_thought"
    description: str = "Think step-by-step about complex problems before answering"
    
    class Parameters(BaseModel):
        question: str = Field(..., description="The question or problem to think through")
    
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

class Document(BaseModel):
    content: str
    embedding: List[float]
    metadata: Dict = {}

# Add a global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Print the full stack trace to terminal
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )

# Tool execution functions
def execute_command(command: str) -> ToolResponse:
    """
    Execute a shell command with safety checks.
    Returns the command output, error (if any), and exit code.
    """
    # Parse the command to get the base command
    try:
        cmd_parts = shlex.split(command)
        base_cmd = cmd_parts[0]
    except Exception as e:
        return ToolResponse(
            output="",
            error=f"Command parsing error: {str(e)}",
            exit_code=1
        )
    
    # Security check - only allow specific commands
    if base_cmd not in ALLOWED_COMMANDS:
        return ToolResponse(
            output="",
            error=f"Command '{base_cmd}' is not allowed for security reasons. Allowed commands: {', '.join(ALLOWED_COMMANDS)}",
            exit_code=1
        )
    
    # Execute the command
    try:
        # Set a timeout to prevent long-running commands
        process = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10  # 10-second timeout
        )
        
        return ToolResponse(
            output=process.stdout,
            error=process.stderr if process.stderr else None,
            exit_code=process.returncode
        )
    except subprocess.TimeoutExpired:
        return ToolResponse(
            output="",
            error="Command execution timed out after 10 seconds",
            exit_code=124
        )
    except Exception as e:
        return ToolResponse(
            output="",
            error=f"Command execution error: {str(e)}",
            exit_code=1
        )

async def execute_rag_search(search_query: str, max_results: int = 5) -> ToolResponse:
    """
    Search for relevant information in the knowledge base.
    Returns the search results or an error message.
    """
    if not supabase:
        return ToolResponse(
            output="",
            error="Knowledge base (Supabase) is not available"
        )
    
    try:
        # Get query embedding from the model
        query_embedding_response = ollama_client.embeddings(
            model="gemma2:27b",
            prompt=search_query
        )
        query_embedding = query_embedding_response['embedding']
        
        # Get relevant contexts
        contexts = await get_relevant_contexts(query_embedding, max_results)
        
        if not contexts:
            return ToolResponse(
                output="No relevant information found in the knowledge base.",
                error=None
            )
        
        # Format the results
        result_text = "Found the following relevant information:\n\n"
        for i, context in enumerate(contexts):
            result_text += f"--- Result {i+1} ---\n{context}\n\n"
        
        return ToolResponse(
            output=result_text,
            error=None
        )
            
    except Exception as e:
        logger.error(f"RAG search error: {str(e)}")
        return ToolResponse(
            output="",
            error=f"Error searching knowledge base: {str(e)}"
        )

def execute_chain_of_thought(question: str) -> ToolResponse:
    """
    Execute the Chain-of-Thought process on a given question.
    Returns a step-by-step reasoning process.
    """
    try:
        # Create a CoT prompting template
        cot_prompt = f"""I'm going to think through this question step-by-step:

Question: {question}

Let me break this down systematically:
1. First, I'll clearly define what the question is asking.
2. I'll identify the key components and variables involved.
3. I'll consider relevant knowledge, principles, or formulas that apply.
4. I'll work through the reasoning process methodically.
5. I'll check my logic for errors or oversights.
6. Finally, I'll arrive at a well-reasoned conclusion.

Let me begin my step-by-step analysis:
"""
        
        # Generate the step-by-step reasoning
        response = ollama_client.generate(
            model="gemma2:27b",
            prompt=cot_prompt,
            options={"temperature": 0.7}  # Slightly higher temperature for more creative thinking
        )
        
        # Format the output
        output = f"Chain-of-Thought Analysis:\n\n{response['response']}"
        
        return ToolResponse(
            output=output,
            error=None
        )
    except Exception as e:
        logger.error(f"Chain-of-Thought execution error: {str(e)}")
        return ToolResponse(
            output="",
            error=f"Error executing Chain-of-Thought reasoning: {str(e)}"
        )

@app.post("/embed")
async def create_embedding(document: Document):
    """Create embeddings for a document and store in Supabase pgvector"""
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase client not available")
    
    try:
        # Log the document properties to help with debugging
        logger.debug(f"Received document with content length: {len(document.content)}")
        logger.debug(f"Embedding length: {len(document.embedding)}")
        logger.debug(f"Metadata: {document.metadata}")
        
        # Validate embedding format - must be a list of floats
        if not isinstance(document.embedding, list):
            raise ValueError(f"Embedding must be a list, got {type(document.embedding)}")
        
        if not all(isinstance(x, float) for x in document.embedding):
            raise ValueError("All embedding values must be floats")
            
        # Store the document and its embedding in Supabase
        data = {
            "content": document.content,
            "embedding": document.embedding,
            "metadata": document.metadata,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Log attempt to insert
        logger.debug(f"Attempting to insert into Supabase table 'documents'")
        
        # Try a more detailed approach to catch Supabase errors
        try:
            result = supabase.table("documents").insert(data).execute()
            logger.debug(f"Insert result: {result}")
            return {"status": "success", "id": result.data[0]["id"]}
        except Exception as inner_e:
            logger.error(f"Supabase insert error details: {str(inner_e)}")
            # Try to get more information about the error
            error_info = str(inner_e)
            if hasattr(inner_e, 'error'):
                error_info += f" Error: {inner_e.error}"
            if hasattr(inner_e, 'message'):
                error_info += f" Message: {inner_e.message}"
            raise ValueError(f"Supabase insert failed: {error_info}")
            
    except Exception as e:
        logger.error(f"Failed to create embedding: {str(e)}")
        # Include the actual error in the response
        raise HTTPException(status_code=500, detail=f"Failed to create embedding: {str(e)}")

async def get_relevant_contexts(query_embedding, max_results):
    """Get relevant contexts from vector database if available"""
    contexts = []
    
    if not supabase:
        logger.warning("Supabase client not available, skipping context retrieval")
        return contexts
    
    try:
        # Check if documents table exists
        try:
            # Simple query to check if table exists
            test_query = supabase.table("documents").select("id").limit(1).execute()
            table_exists = True
        except Exception as e:
            if "relation \"documents\" does not exist" in str(e):
                logger.warning("Documents table does not exist")
                return contexts
            raise  # Re-raise if it's a different error
        
        # Check if match_documents function exists
        try:
            query = supabase.postgrest.rpc(
                "match_documents",
                {
                    "query_embedding": query_embedding,
                    "match_count": max_results
                }
            ).execute()
            
            # Extract relevant contexts
            contexts = [doc['content'] for doc in query.data]
        except Exception as e:
            if "function public.match_documents" in str(e) and "not found" in str(e):
                logger.warning("match_documents function does not exist")
                return contexts
            raise  # Re-raise if it's a different error
            
    except Exception as e:
        logger.warning(f"Error retrieving contexts: {str(e)}")
        # Continue without contexts rather than failing
        
    return contexts

def format_tool_specifications():
    """Format tool specifications for the LLM prompt"""
    tools = {
        "command_execution": {
            "description": "Execute a shell command and return its output",
            "parameters": {
                "command": "The command to execute"
            },
            "allowed_commands": ALLOWED_COMMANDS
        },
        "rag_search": {
            "description": "Search for relevant information in the knowledge base",
            "parameters": {
                "query": "The search query to find relevant information",
                "max_results": "Maximum number of results to return (default: 5)"
            }
        },
        "chain_of_thought": {
            "description": "Think step-by-step about complex problems before answering",
            "parameters": {
                "question": "The question or problem to think through"
            }
        }
    }
    return json.dumps(tools, indent=2)

@app.post("/query")
async def query(input_data: QueryInput):
    """Query the model with optional tool usage"""
    # Base prompt is just the user's query
    base_prompt = input_data.query
    
    # Add tool specifications if tools are allowed
    if input_data.allow_tools:
        tool_specs = format_tool_specifications()
        prompt = f"""You have access to the following tools:
                            
                    {tool_specs}

                    When you need to use a tool, respond with JSON in the following format:
                    {{
                        "thoughts": "your reasoning about what tool to use and why",
                        "tool_calls": [
                            {{
                                "tool": "tool_name",
                                "parameters": {{
                                    "param1": "value1",
                                    "param2": "value2"
                                }}
                            }}
                        ]
                    }}

                    You should use the rag_search tool when you need to find information that might be in the knowledge base.
                    You should use the command_execution tool when you need to run a command on the system.
                    You should use the chain_of_thought tool when faced with complex problems requiring step-by-step reasoning.

                    If you don't need to use a tool, just respond normally to the query.

                    Now, please respond to the following query:
                    {base_prompt}"""
    else:
        prompt = base_prompt
    
    try:
        # First pass - generate initial response from the LLM
        response = ollama_client.generate(
            model="gemma2:27b",
            prompt=prompt,
            options={"temperature": input_data.temperature}
        )
        
        model_response = response['response']
        tool_response = None
        
        # Check if the response contains a tool call
        if input_data.allow_tools:
            try:
                # Try to find JSON pattern using simple heuristic
                if "{" in model_response and "}" in model_response:
                    json_start = model_response.find("{")
                    json_end = model_response.rfind("}") + 1
                    json_str = model_response[json_start:json_end]
                    
                    try:
                        parsed = json.loads(json_str)
                        
                        # Check if this has the expected structure
                        if "tool_calls" in parsed and isinstance(parsed["tool_calls"], list) and len(parsed["tool_calls"]) > 0:
                            # Extract the tool call
                            tool_call = parsed["tool_calls"][0]
                            tool_name = tool_call.get("tool")
                            params = tool_call.get("parameters", {})
                            
                            # Execute the appropriate tool
                            if tool_name == "command_execution":
                                command = params.get("command")
                                if command:
                                    result = execute_command(command)
                                    tool_response = {
                                        "tool": "command_execution",
                                        "parameters": {"command": command},
                                        "result": result.dict()
                                    }
                                    
                                    # Generate a follow-up response with the tool results
                                    followup_prompt = f"""You previously decided to use the command_execution tool with the command: {command}

                                                        The tool returned the following result:
                                                        Exit code: {result.exit_code}
                                                        Output: {result.output or "No output"}
                                                        Error: {result.error or "No error"}

                                                        Based on this result, please provide a final response to the user's original query:
                                                        {input_data.query}"""

                                    followup_response = ollama_client.generate(
                                        model="gemma2:27b",
                                        prompt=followup_prompt,
                                        options={"temperature": input_data.temperature}
                                    )
                                    
                                    model_response = followup_response['response']
                            
                            elif tool_name == "rag_search":
                                search_query = params.get("query")
                                max_results = params.get("max_results", 5)
                                
                                if search_query:
                                    result = await execute_rag_search(search_query, max_results)
                                    tool_response = {
                                        "tool": "rag_search",
                                        "parameters": {
                                            "query": search_query,
                                            "max_results": max_results
                                        },
                                        "result": result.dict()
                                    }
                                    
                                    # Generate a follow-up response with the search results
                                    followup_prompt = f"""You previously decided to search the knowledge base with query: "{search_query}"

                                                        The search returned the following results:
                                                        {result.output or "No results found"}
                                                        {f"Error: {result.error}" if result.error else ""}

                                                        Based on these search results, please provide a final response to the user's original query:
                                                        {input_data.query}

                                                        Remember to incorporate the relevant information from the search results, but don't mention that you performed a search unless it's directly relevant to answering the query."""

                                    followup_response = ollama_client.generate(
                                        model="gemma2:27b",
                                        prompt=followup_prompt,
                                        options={"temperature": input_data.temperature}
                                    )
                                    
                                    model_response = followup_response['response']
                            
                            elif tool_name == "chain_of_thought":
                                question = params.get("question")
                                
                                if question:
                                    result = execute_chain_of_thought(question)
                                    tool_response = {
                                        "tool": "chain_of_thought",
                                        "parameters": {
                                            "question": question
                                        },
                                        "result": result.dict()
                                    }
                                    
                                    # Generate a follow-up response with the reasoning results
                                    followup_prompt = f"""You previously decided to use step-by-step reasoning to think through this question: "{question}"

                                                        Here is the step-by-step analysis:
                                                        {result.output or "No analysis generated"}
                                                        {f"Error: {result.error}" if result.error else ""}

                                                        Based on this analysis, please provide a final response to the user's original query:
                                                        {input_data.query}

                                                        You can incorporate parts of the step-by-step reasoning in your answer to show your work."""

                                    followup_response = ollama_client.generate(
                                        model="gemma2:27b",
                                        prompt=followup_prompt,
                                        options={"temperature": input_data.temperature}
                                    )
                                    
                                    model_response = followup_response['response']
                    except json.JSONDecodeError:
                        # Not valid JSON, continue with normal response
                        pass
            except Exception as e:
                logger.warning(f"Error processing tool call: {str(e)}")
        
        # Return final response
        return {
            "response": model_response,
            "tool_used": tool_response is not None,
            "tool_details": tool_response
        }
    
    except Exception as e:
        logger.error(f"Error during LLM inference: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM inference failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "api": "healthy",
        "ollama": "unknown",
        "supabase": "not configured" if not supabase else "unknown",
        "tools": {
            "command_execution": "available",
            "rag_search": "available" if supabase else "unavailable (requires Supabase)",
            "chain_of_thought": "available"
        }
    }
    
    # Check Ollama
    try:
        ollama_client.list()
        status["ollama"] = "healthy"
    except Exception as e:
        status["ollama"] = f"error: {str(e)}"
    
    # Check Supabase if configured
    if supabase:
        try:
            # Simple query to check connection
            supabase.from_("").select("*", count="exact").limit(0).execute()
            status["supabase"] = "healthy"
        except Exception as e:
            status["supabase"] = f"error: {str(e)}"
    
    return status

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")