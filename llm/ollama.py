import logging
from typing import AsyncGenerator
from ollama import Client as OllamaClient
from config import DEFAULT_MODEL, TOOL_MODELS

logger = logging.getLogger(__name__)

# Initialize Ollama client
ollama_client = OllamaClient(host="http://localhost:11434")

async def generate_response_stream(prompt, model=None, temperature=0.7) -> AsyncGenerator[str, None]:
    """Generate a streaming response from the LLM"""
    try:
        model_to_use = model or DEFAULT_MODEL
        # Convert the regular generator to an async generator
        for chunk in ollama_client.generate(
            model=model_to_use,
            prompt=prompt,
            options={"temperature": temperature},
            stream=True
        ):
            if "response" in chunk:
                yield chunk["response"]
    except Exception as e:
        logger.error(f"Error generating streaming response: {str(e)}")
        raise

def generate_response(prompt, model=None, temperature=0.7):
    """Generate a complete response from the LLM (non-streaming)"""
    try:
        model_to_use = model or DEFAULT_MODEL
        response = ollama_client.generate(
            model=model_to_use,
            prompt=prompt,
            options={"temperature": temperature},
        )
        return response["response"]
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise

def get_embedding(text, model=None):
    """Get embedding for text"""
    try:
        embedding_model = model or TOOL_MODELS.get("embeddings", DEFAULT_MODEL)
        response = ollama_client.embeddings(
            model=embedding_model, prompt=text
        )
        return response["embedding"]
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise


def list_available_models():
    """List available models from Ollama"""
    try:
        models = ollama_client.list()
        return [model["name"] for model in models.get("models", [])]
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return []


def check_connection():
    """Check if Ollama connection is healthy"""
    try:
        models = ollama_client.list()
        return len(models.get("models", [])) > 0
    except Exception as e:
        logger.error(f"Ollama connection error: {str(e)}")
        return False