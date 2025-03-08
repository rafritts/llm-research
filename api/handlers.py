import logging
import json
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from models.schema import QueryInput, ModelConfigInput, Document
from config import DEFAULT_MODEL, TOOL_MODELS
from llm.ollama import generate_response, get_embedding, list_available_models, check_connection as check_ollama
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

    try:
        # First pass - generate initial response from the LLM
        model_response = generate_response(
            prompt, 
            model=query_model, 
            temperature=input_data.temperature
        )
        
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
                        if (
                            "tool_calls" in parsed
                            and isinstance(parsed["tool_calls"], list)
                            and len(parsed["tool_calls"]) > 0
                        ):
                            # Extract the tool call
                            tool_call = parsed["tool_calls"][0]
                            tool_name = tool_call.get("tool")
                            params = tool_call.get("parameters", {})

                            # Execute the appropriate tool
                            if tool_name == "command_execution":
                                result = await CommandExecutionTool.execute(params)
                                tool_response = {
                                    "tool": "command_execution",
                                    "parameters": params,
                                    "result": result.dict(),
                                }

                                # Get the model for command execution followup
                                cmd_model = TOOL_MODELS.get("command_execution", query_model)
                                
                                # Generate a follow-up response with the tool results
                                followup_prompt = f"""You previously decided to use the command_execution tool with the command: {params.get('command')}

                                                    The tool returned the following result:
                                                    Exit code: {result.exit_code}
                                                    Output: {result.output or "No output"}
                                                    Error: {result.error or "No error"}

                                                    Based on this result, please provide a final response to the user's original query:
                                                    {input_data.query}"""

                                model_response = generate_response(
                                    followup_prompt,
                                    model=cmd_model,
                                    temperature=input_data.temperature
                                )

                            elif tool_name == "rag_search":
                                result = await RAGSearchTool.execute(params)
                                tool_response = {
                                    "tool": "rag_search",
                                    "parameters": params,
                                    "result": result.dict(),
                                }

                                # Get the model for RAG search followup
                                rag_model = TOOL_MODELS.get("rag_search", query_model)
                                
                                # Generate a follow-up response with the search results
                                followup_prompt = f"""You previously decided to search the knowledge base with query: "{params.get('query')}"

                                                    The search returned the following results:
                                                    {result.output or "No results found"}
                                                    {f"Error: {result.error}" if result.error else ""}

                                                    Based on these search results, please provide a final response to the user's original query:
                                                    {input_data.query}

                                                    Remember to incorporate the relevant information from the search results, but don't mention that you performed a search unless it's directly relevant to answering the query."""

                                model_response = generate_response(
                                    followup_prompt,
                                    model=rag_model,
                                    temperature=input_data.temperature
                                )

                            elif tool_name == "chain_of_thought":
                                result = await ChainOfThoughtTool.execute(params)
                                tool_response = {
                                    "tool": "chain_of_thought",
                                    "parameters": params,
                                    "result": result.dict(),
                                }

                                # Get the model for chain of thought followup
                                cot_model = TOOL_MODELS.get("chain_of_thought", query_model)
                                
                                # Generate a follow-up response with the reasoning results
                                followup_prompt = f"""You previously decided to use step-by-step reasoning to think through this question: "{params.get('question')}"

                                                    Here is the step-by-step analysis:
                                                    {result.output or "No analysis generated"}
                                                    {f"Error: {result.error}" if result.error else ""}

                                                    Based on this analysis, please provide a final response to the user's original query:
                                                    {input_data.query}

                                                    You can incorporate parts of the step-by-step reasoning in your answer to show your work."""

                                model_response = generate_response(
                                    followup_prompt,
                                    model=cot_model,
                                    temperature=input_data.temperature
                                )
                    except json.JSONDecodeError:
                        # Not valid JSON, continue with normal response
                        pass
            except Exception as e:
                logger.warning(f"Error processing tool call: {str(e)}")

        # Return final response
        return {
            "response": model_response,
            "tool_used": tool_response is not None,
            "tool_details": tool_response,
            "model_used": query_model,
        }

    except Exception as e:
        logger.error(f"Error during LLM inference: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM inference failed: {str(e)}")

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