import logging
import json
import asyncio
from typing import AsyncGenerator
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from models.schema import QueryInput, ModelConfigInput, Document
from config import DEFAULT_MODEL, TOOL_MODELS
from llm.ollama import generate_response, generate_response_stream, get_embedding, list_available_models, check_connection as check_ollama
from database.supabase import supabase, store_document, check_connection as check_supabase
from tools.base import format_tool_specifications
from tools.command_execution import CommandExecutionTool
from tools.rag_search import RAGSearchTool
from tools.chain_of_thought import ChainOfThoughtTool

logger = logging.getLogger(__name__)

async def handle_update_model_config(config: ModelConfigInput):
    """Update the model configuration"""
    global DEFAULT_MODEL, TOOL_MODELS
    
    try:
        # Validate that the models exist in Ollama
        available_models = list_available_models()
        
        if config.default_model not in available_models:
            return JSONResponse(
                status_code=400, 
                content={"detail": f"Model {config.default_model} not available in Ollama"}
            )
        
        # Update default model
        DEFAULT_MODEL = config.default_model
        
        # Update tool models
        for tool, model in config.tool_models.items():
            if model not in available_models:
                return JSONResponse(
                    status_code=400, 
                    content={"detail": f"Model {model} not available in Ollama"}
                )
            TOOL_MODELS[tool] = model
        
        return {
            "status": "success",
            "default_model": DEFAULT_MODEL,
            "tool_models": TOOL_MODELS
        }
    except Exception as e:
        logger.error(f"Failed to update model configuration: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to update model configuration: {str(e)}"
        )

async def handle_get_model_config():
    """Get the current model configuration"""
    try:
        # Fetch available models from Ollama
        available_models = list_available_models()
        
        return {
            "default_model": DEFAULT_MODEL,
            "tool_models": TOOL_MODELS,
            "available_models": available_models
        }
    except Exception as e:
        logger.error(f"Failed to get model configuration: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get model configuration: {str(e)}"
        )

async def handle_query(input_data: QueryInput):
    """Query the model with optional tool usage"""
    # Determine which model to use for this query
    query_model = input_data.model or DEFAULT_MODEL
    
    # If chain of thought is requested, automatically wrap the query
    if input_data.use_chain_of_thought:
        base_prompt = f"""Please use the chain_of_thought tool to analyze this query thoroughly before responding:
                         {input_data.query}"""
    else:
        base_prompt = input_data.query

    # Enable tools if chain of thought is requested
    if input_data.use_chain_of_thought:
        input_data.allow_tools = True

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

    async def generate_streaming_response() -> AsyncGenerator[str, None]:
        try:
            first_chunk = True
            async for chunk in generate_response_stream(
                prompt,
                model=query_model,
                temperature=input_data.temperature
            ):
                # For the first chunk, check if it might be a tool call
                if first_chunk and input_data.allow_tools and "{" in chunk:
                    # Buffer the response until we can determine if it's JSON
                    buffer = chunk
                    async for next_chunk in generate_response_stream(
                        prompt,
                        model=query_model,
                        temperature=input_data.temperature
                    ):
                        buffer += next_chunk
                        if "}" in buffer:
                            break
                    
                    try:
                        # Try to parse as JSON
                        json_start = buffer.find("{")
                        json_end = buffer.rfind("}") + 1
                        json_str = buffer[json_start:json_end]
                        parsed = json.loads(json_str)
                        
                        if "tool_calls" in parsed and isinstance(parsed["tool_calls"], list):
                            # Handle tool call
                            tool_call = parsed["tool_calls"][0]
                            tool_name = tool_call.get("tool")
                            params = tool_call.get("parameters", {})
                            
                            if tool_name in ["command_execution", "rag_search", "chain_of_thought"]:
                                # Execute tool and get follow-up response
                                if tool_name == "command_execution":
                                    tool = CommandExecutionTool
                                    model = TOOL_MODELS.get("command_execution", query_model)
                                elif tool_name == "rag_search":
                                    tool = RAGSearchTool
                                    model = TOOL_MODELS.get("rag_search", query_model)
                                else:  # chain_of_thought
                                    tool = ChainOfThoughtTool
                                    model = TOOL_MODELS.get("chain_of_thought", query_model)
                                
                                result = await tool.execute(params)
                                
                                # Generate follow-up response with tool results
                                followup_prompt = f"""Based on the tool results:
                                                    {result.output or 'No output'}
                                                    {f"Error: {result.error}" if result.error else ""}
                                                    
                                                    Please provide a final response to the original query:
                                                    {input_data.query}"""
                                
                                async for chunk in generate_response_stream(
                                    followup_prompt,
                                    model=model,
                                    temperature=input_data.temperature
                                ):
                                    yield chunk
                                return
                            
                    except json.JSONDecodeError:
                        # Not valid JSON, proceed with normal streaming
                        yield buffer
                
                first_chunk = False
                yield chunk

        except Exception as e:
            logger.error(f"Error during streaming LLM inference: {str(e)}")
            raise HTTPException(status_code=500, detail=f"LLM inference failed: {str(e)}")

    return StreamingResponse(
        generate_streaming_response(),
        media_type="text/event-stream"
    )

async def handle_create_embedding(document: Document):
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
            raise ValueError(
                f"Embedding must be a list, got {type(document.embedding)}"
            )

        if not all(isinstance(x, float) for x in document.embedding):
            raise ValueError("All embedding values must be floats")

        # Store the document and its embedding in Supabase
        doc_id = store_document(document.content, document.embedding, document.metadata)
        
        if not doc_id:
            raise HTTPException(
                status_code=500, 
                detail="Failed to store document in database"
            )
            
        return {"status": "success", "id": doc_id}

    except Exception as e:
        logger.error(f"Failed to create embedding: {str(e)}")
        # Include the actual error in the response
        raise HTTPException(
            status_code=500, detail=f"Failed to create embedding: {str(e)}"
        )

async def handle_create_text_embedding(text: str):
    """Create an embedding for a text string using the configured embeddings model"""
    try:
        # Use the embeddings model to generate the embedding
        embedding_model = TOOL_MODELS.get("embeddings", DEFAULT_MODEL)
        embedding = get_embedding(text, model=embedding_model)
        
        return {
            "status": "success", 
            "embedding": embedding,
            "model_used": embedding_model
        }
    except Exception as e:
        logger.error(f"Failed to create text embedding: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create text embedding: {str(e)}"
        )

async def handle_health_check():
    """Health check endpoint"""
    status = {
        "api": "healthy",
        "ollama": "unknown",
        "supabase": "not configured" if not supabase else "unknown",
        "models": {
            "default": DEFAULT_MODEL,
            "tools": TOOL_MODELS,
        },
        "tools": {
            "command_execution": "available",
            "rag_search": (
                "available" if supabase else "unavailable (requires Supabase)"
            ),
            "chain_of_thought": "available",
        },
    }

    # Check Ollama
    if check_ollama():
        status["ollama"] = "healthy"
        status["available_models"] = list_available_models()
    else:
        status["ollama"] = "unhealthy"

    # Check Supabase if configured
    if supabase:
        status["supabase"] = "healthy" if check_supabase() else "unhealthy"

    return status