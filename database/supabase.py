import logging
from datetime import datetime
from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY

logger = logging.getLogger(__name__)

# Initialize Supabase client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    logger.warning(f"Failed to initialize Supabase client: {str(e)}")
    supabase = None


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
        except Exception as e:
            if 'relation "documents" does not exist' in str(e):
                logger.warning("Documents table does not exist")
                return contexts
            raise  # Re-raise if it's a different error

        # Check if match_documents function exists
        try:
            query = supabase.postgrest.rpc(
                "match_documents_precision",
                {"query_embedding": query_embedding, "match_count": max_results, "match_threshold": 0.35},
            ).execute()

            # Extract relevant contexts
            contexts = [doc["content"] for doc in query.data]
        except Exception as e:
            if "function public.match_documents" in str(e) and "not found" in str(e):
                logger.warning("match_documents function does not exist")
                return contexts
            raise  # Re-raise if it's a different error

    except Exception as e:
        logger.warning(f"Error retrieving contexts: {str(e)}")
        # Continue without contexts rather than failing

    return contexts


def store_document(content, embedding, metadata=None):
    """Store document and its embedding in Supabase"""
    if not supabase:
        logger.warning("Supabase client not available, skipping document storage")
        return None
    
    try:
        data = {
            "content": content,
            "embedding": embedding,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
        }
        
        result = supabase.table("documents").insert(data).execute()
        return result.data[0]["id"] if result.data else None
    except Exception as e:
        logger.error(f"Failed to store document: {str(e)}")
        return None


def check_connection():
    """Check if Supabase connection is healthy"""
    if not supabase:
        return False
        
    try:
        # Simple query to check connection
        supabase.from_("").select("*", count="exact").limit(0).execute()
        return True
    except Exception as e:
        logger.error(f"Supabase connection error: {str(e)}")
        return False