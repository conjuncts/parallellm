import logging

from pydantic import BaseModel
from parallellm.core.gateway import ParalleLLM
from dotenv import load_dotenv

load_dotenv()


class MyModel(BaseModel):
    final_answer: str


tools_openai = [
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

tools_anthropic = [
    {
        "name": "count_files",
        "description": "Count the number of files in a directory.",
        "input_schema": {
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

tools_google = [x.copy() for x in tools_openai]
[x.pop("type") for x in tools_google]

with ParalleLLM.resume_directory(
    ".pllm/simplest-tool",
    # ".temp",
    provider="google",  #
    strategy="sync",
    log_level=logging.DEBUG,
    user_confirmation=True,
    # ignore_cache=True,
) as pllm:
    with pllm.agent(dashboard=True) as dash:
        # Structured output
        # resp = dash.ask_llm(
        # "Please name a power of 3.", hash_by=["llm"], text_format=MyModel
        # )

        # Tools
        resp = dash.ask_llm(
            "How many files are in '~/examples'?",
            hash_by=["llm"],
            tools=tools_google,
        )

        dash.print(resp.resolve())
        for name, args, call_id in resp.resolve_tool_calls():
            dash.print(f"Tool call: `{name}` with args {args} call_id {call_id}")

        # Should respond by putting it in:
        # - function_call_output (openai)
        # - tool_result (anthropic)

# pllm.persist()
