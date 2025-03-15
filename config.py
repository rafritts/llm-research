import os
import logging
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Model configuration
class ModelNames(str, Enum):
    GEMMA = "gemma3:27b"
    DEEPSEEKR1 = "deepseek-r1:32b"

DEFAULT_MODEL = ModelNames.GEMMA
TOOL_MODELS = {
    "command_execution": ModelNames.GEMMA,
    "rag_search": ModelNames.GEMMA, 
    "chain_of_thought": ModelNames.GEMMA,
    "embeddings": ModelNames.GEMMA
}

# Available tools configuration
ALLOWED_COMMANDS = [
    "ls", "pwd", "cat", "grep", "find", "echo", "date", 
    "wc", "head", "tail", "df", "du", "ps",
]