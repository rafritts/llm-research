import logging
from typing import AsyncGenerator
from fastapi import HTTPException
from llm.ollama import generate_response_stream

logger = logging.getLogger(__name__)

class AsyncStreamingUtils:
    @staticmethod
    async def handle_streaming_response(
        prompt: str,
        query_model: str,
        temperature: float,
        allow_tools: bool,
        original_query: str
    ) -> AsyncGenerator[str, None]:
        """Stream response chunks from the LLM"""
        try:
            async for chunk in generate_response_stream(
                prompt, model=query_model, temperature=temperature
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Error during streaming LLM inference: {str(e)}")
            raise HTTPException(status_code=500, detail=f"LLM inference failed: {str(e)}")