import json
import logging
from typing import AsyncGenerator, Any, Dict
from fastapi import HTTPException
from llm.ollama import generate_response_stream
from tools.command_execution import CommandExecutionTool

logger = logging.getLogger(__name__)

class AsyncStreamingUtils:
    @staticmethod
    async def handle_streaming_response(
        prompt: str,
        query_model: str,
        temperature: float,
        allow_tools: bool,
        original_query: str
    ) -> AsyncGenerator[str, None]:
        try:
            first_chunk = True
            async for chunk in generate_response_stream(
                prompt, model=query_model, temperature=temperature
            ):
                # For the first chunk, check if it might be a tool call
                if first_chunk and allow_tools and "{" in chunk:
                    buffer = chunk
                    async for next_chunk in generate_response_stream(
                        prompt, model=query_model, temperature=temperature
                    ):
                        buffer += next_chunk
                        if "}" in buffer:
                            break

                    try:
                        json_start = buffer.find("{")
                        json_end = buffer.rfind("}") + 1
                        json_str = buffer[json_start:json_end]
                        parsed = json.loads(json_str)

                        if "tool_calls" in parsed and isinstance(parsed["tool_calls"], list):
                            tool_call = parsed["tool_calls"][0]
                            tool_name = tool_call.get("tool")
                            params = tool_call.get("parameters", {})

                            if tool_name == "command_execution":
                                tool = CommandExecutionTool()
                                result = await tool.execute(params)
                                followup_prompt = f"""Based on the command results:
                                                    {result.output or 'No output'}
                                                    {f"Error: {result.error}" if result.error else ""}

                                                    Please provide a final response to: {original_query}"""

                                async for chunk in generate_response_stream(
                                    followup_prompt,
                                    model=query_model,
                                    temperature=temperature,
                                ):
                                    yield chunk
                                return

                    except json.JSONDecodeError:
                        yield buffer

                first_chunk = False
                yield chunk
        except Exception as e:
            logger.error(f"Error during streaming LLM inference: {str(e)}")
            raise HTTPException(status_code=500, detail=f"LLM inference failed: {str(e)}")