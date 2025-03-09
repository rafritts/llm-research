from tools.rag_search import RAGSearchTool

async def generate_rag_response(query: str, max_results: int = 5) -> str:
    """
    Generate a RAG-enhanced prompt by searching the knowledge base and combining with the query
    
    Args:
        query: The user's query
        max_results: Maximum number of search results to include
        
    Returns:
        str: The RAG-enhanced prompt
    """
    rag_tool = RAGSearchTool()
    result = await rag_tool.execute({"query": query, "max_results": max_results})
    
    return f"""Based on the following context from the knowledge base:
            {result.output}

            Please answer the original query:
            {query}

            If the context is not relevant to the original query, you must ignore it and answer the query directly. 
            Do not reference the context directly as 'the context' in your response."""