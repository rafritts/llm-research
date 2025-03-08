import uvicorn
from fastapi import FastAPI
from api.routes import setup_routes

# Initialize FastAPI app
app = FastAPI(
    title="LLM with Tool-Based RAG, Command Execution, and Chain-of-Thought",
    root_path_in_servers=False,
    max_request_body_size=1024 * 1024,  # 1MB
)

# Setup routes
setup_routes(app)

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0",
        port=8000, 
        log_level="debug"
    )