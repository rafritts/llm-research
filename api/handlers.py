import logging
from fastapi.responses import StreamingResponse
from models.schema import QueryInput, Document
from config import DEFAULT_MODEL
from tools.base import format_tool_specifications
from service.chain_of_thought_service import generate_chain_of_thought_response
from service.embeddings import create_document_embedding, create_text_embedding
from service.rag_service import generate_rag_response
from .util import AsyncStreamingUtils

logger = logging.getLogger(__name__)


async def handle_query(input_data: QueryInput):
    """Query the model with optional RAG, CoT, or tool usage"""
    query_model = input_data.model or DEFAULT_MODEL

    if input_data.use_rag:
        # Use RAG service
        prompt = await generate_rag_response(input_data.query)

    elif input_data.use_cot:
        # Use Chain of Thought via service
        prompt = await generate_chain_of_thought_response(input_data.query)

    else:
        # Regular query with optional tool access
        tool_specs = format_tool_specifications() if input_data.allow_tools else ""
        prompt = f"""{tool_specs if tool_specs else ""}
                        {input_data.query}"""
        if input_data.allow_tools:
            prompt = f"""When you need to use a tool, respond with JSON in the following format:
                        {{
                            "thoughts": "your reasoning about what tool to use and why",
                            "tool_calls": [
                                {{
                                    "tool": "tool_name",
                                    "parameters": {{
                                        "param1": "value1"
                                    }}
                                }}
                            ]
                        }}
                        You can only use one tool. If you don't need to use a tool, just respond normally.

                        {prompt}"""

    return StreamingResponse(
        AsyncStreamingUtils.handle_streaming_response(
            prompt=prompt,
            query_model=query_model,
            temperature=input_data.temperature,
            allow_tools=input_data.allow_tools,
            original_query=input_data.query,
        ),
        media_type="text/event-stream",
    )


async def handle_create_embedding(document: Document):
    """Create embeddings for a document and store in Supabase pgvector"""
    return await create_document_embedding(document)


async def handle_create_text_embedding(text: str):
    """Create an embedding for a text string using the configured embeddings model"""
    return await create_text_embedding(text)
