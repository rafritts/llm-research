import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import traceback
import sys
from api.handlers import (
    handle_query, 
    handle_create_embedding, 
    handle_create_text_embedding
)

logger = logging.getLogger(__name__)

def setup_routes(app: FastAPI):
    """Setup all routes for the application"""
    
    # Add a global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        # Print the full stack trace to terminal
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        logger.error(f"Unhandled exception: {str(exc)}")
        return JSONResponse(status_code=500, content={"detail": str(exc)})
    
    # Query endpoint
    app.post("/query")(handle_query)
    
    # Embedding endpoints
    app.post("/embed")(handle_create_embedding)
    app.post("/embed-text")(handle_create_text_embedding)
    