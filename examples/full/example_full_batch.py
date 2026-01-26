import logging

from pydantic import BaseModel
from parallellm.core.gateway import ParalleLLM
from dotenv import load_dotenv

from parallellm.tools.server import WebSearchTool

load_dotenv()


class MyModel(BaseModel):
    final_answer: str


tools = [
    {
        "type": "function",
        "name": "count_files",
        "description": "Count the number of files in a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "The path to the directory to count files in.",
                },
            },
            "required": ["directory"],
        },
    }
]

with ParalleLLM.resume_directory(
    ".pllm/example/batch",
    provider="google",
    strategy="batch",
    log_level=logging.DEBUG,
    # ignore_cache=True,
) as pllm:
    with pllm.agent(dashboard=True) as dash:
        resp1 = dash.ask_llm("Please name a power of 3.", hash_by=["llm"])
        resp2 = dash.ask_llm(
            "In 1 sentence, what is AAPL's current price?",
            # llm="claude-haiku-4-5-20251001",
            tools=[WebSearchTool()],
            hash_by=["llm"],
        )
        resp3 = dash.ask_llm(
            "How many files are in ~/examples? Give the final answer in words.",
            hash_by=["llm"],
            tools=tools,
        )
        resp4 = dash.ask_llm(
            "What is the capital of France?", hash_by=["llm"], text_format=MyModel
        )

        for resp in [resp1, resp2, resp3, resp4]:
            if fcs := resp.resolve_function_calls():
                for fc in fcs:
                    dash.print(f"function_call {fc.name} {fc.args} {fc.call_id}")
            dash.print(resp.resolve())
