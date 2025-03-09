import logging
from fastapi import HTTPException
from models.schema import Document
from config import DEFAULT_MODEL, TOOL_MODELS
from llm.ollama import get_embedding
from database.supabase import supabase, store_document

logger = logging.getLogger(__name__)

async def create_document_embedding(document: Document):
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
        doc_id = store_document(document.content, document.embedding, document.metadata)

        if not doc_id:
            raise HTTPException(status_code=500, detail="Failed to store document in database")

        return {"status": "success", "id": doc_id}

    except Exception as e:
        logger.error(f"Failed to create embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create embedding: {str(e)}")

async def create_text_embedding(text: str):
    """Create an embedding for a text string using the configured embeddings model"""
    try:
        # Use the embeddings model to generate the embedding
        embedding_model = TOOL_MODELS.get("embeddings", DEFAULT_MODEL)
        embedding = get_embedding(text, model=embedding_model)

        return {
            "status": "success",
            "embedding": embedding,
            "model_used": embedding_model,
        }
    except Exception as e:
        logger.error(f"Failed to create text embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create text embedding: {str(e)}")