from tools.chain_of_thought import ChainOfThoughtTool

async def generate_chain_of_thought_response(question: str) -> str:
    """
    Generate a chain of thought response for the given question
    """
    print("Generating chain of thought response")
    cot_tool = ChainOfThoughtTool()
    result = await cot_tool.execute({"question": question})
    
    prompt = f"""Based on this step-by-step reasoning:
                {result.output}

                Please provide a final answer to:
                {question}"""
    
    return prompt