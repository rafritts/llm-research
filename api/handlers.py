import logging
from fastapi.responses import StreamingResponse
from models.schema import QueryInput, Document
from config import DEFAULT_MODEL
from service.chain_of_thought_service import generate_chain_of_thought_response
from service.command_execution_service import generate_tool_execution_prompt
from service.embeddings import create_document_embedding, create_text_embedding
from service.rag_service import generate_rag_response
from llm.ollama import generate_response_stream

logger = logging.getLogger(__name__)


async def handle_query(input_data: QueryInput):
    """Query the model with optional RAG, CoT, or tool usage"""
    query_model = input_data.model or DEFAULT_MODEL

    if input_data.use_rag:
        # Use RAG service
        prompt = await generate_rag_response(input_data.query)
        response_stream = generate_response_stream(prompt, model=query_model, temperature=input_data.temperature)

    elif input_data.use_cot:
        # Use Chain of Thought via service
        prompt = await generate_chain_of_thought_response(input_data.query)
        response_stream = generate_response_stream(prompt, model=query_model, temperature=input_data.temperature)

    else:
        # Regular query with optional tool access
        if input_data.allow_tools:
            response_stream = generate_tool_execution_prompt(
                query=input_data.query,
                model=query_model,
                temperature=input_data.temperature
            )
        else:
            response_stream = generate_response_stream(
                input_data.query,
                model=query_model,
                temperature=input_data.temperature
            )

    return StreamingResponse(
        response_stream,
        media_type="text/event-stream",
    )


async def handle_create_embedding(document: Document):
    """Create embeddings for a document and store in Supabase pgvector"""
    return await create_document_embedding(document)


async def handle_create_text_embedding(text: str):
    """Create an embedding for a text string using the configured embeddings model"""
    return await create_text_embedding(text)
