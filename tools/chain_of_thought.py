import logging
from typing import Dict, Any
from models.schema import ToolResponse
from tools.base import BaseTool
from config import TOOL_MODELS, DEFAULT_MODEL
from llm.ollama import generate_response

logger = logging.getLogger(__name__)

class ChainOfThoughtTool(BaseTool):
    name = "chain_of_thought"
    description = "Think step-by-step about complex problems before answering"
    
    @classmethod
    async def execute(cls, parameters: Dict[str, Any]):
        """Execute the Chain-of-Thought process on a given question"""
        question = parameters.get("question", "")
        
        try:
            # Get the appropriate model for chain-of-thought
            cot_model = TOOL_MODELS.get("chain_of_thought", DEFAULT_MODEL)
            
            # Create a CoT prompting template
            cot_prompt = f"""I'm going to think through this question step-by-step:

                            Question: {question}

                            Let me break this down systematically:
                            1. First, I'll clearly define what the question is asking.
                            2. I'll identify the key components and variables involved.
                            3. I'll consider relevant knowledge, principles, or formulas that apply.
                            4. I'll work through the reasoning process methodically.
                            5. I'll check my logic for errors or oversights.
                            6. Finally, I'll arrive at a well-reasoned conclusion.

                            Let me begin my step-by-step analysis:
                            """

            # Generate the step-by-step reasoning
            response = generate_response(cot_prompt, model=cot_model, temperature=0.7)
            
            # Format the output
            output = f"Chain-of-Thought Analysis:\n\n{response}"

            # Log the output
            logger.info(f"Chain-of-Thought output: {output}")

            return ToolResponse(output=output, error=None)
        except Exception as e:
            logger.error(f"Chain-of-Thought execution error: {str(e)}")
            return ToolResponse(
                output="", error=f"Error executing Chain-of-Thought reasoning: {str(e)}"
            )