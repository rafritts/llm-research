import logging
from typing import Dict, Any
from models.schema import ToolResponse
from tools.base import BaseTool
from llm.ollama import get_embedding
from database.supabase import get_relevant_contexts

logger = logging.getLogger(__name__)

class RAGSearchTool(BaseTool):
    name = "rag_search"
    description = "Search for relevant information in the knowledge base"
    
    @classmethod
    async def execute(cls, parameters: Dict[str, Any]):
        """Search for relevant information in the knowledge base"""
        search_query = parameters.get("query", "")
        max_results = parameters.get("max_results", 5)
        
        try:
            # Get query embedding from the model
            query_embedding = get_embedding(search_query)
            
            # Get relevant contexts
            contexts = await get_relevant_contexts(query_embedding, max_results)

            if not contexts:
                return ToolResponse(
                    output="No relevant information found in the knowledge base.",
                    error=None,
                )

            # Format the results
            result_text = "Found the following relevant information:\n\n"
            for i, context in enumerate(contexts):
                result_text += f"--- Result {i+1} ---\n{context}\n\n"

            return ToolResponse(output=result_text, error=None)

        except Exception as e:
            logger.error(f"RAG search error: {str(e)}")
            return ToolResponse(
                output="", error=f"Error searching knowledge base: {str(e)}"
            )