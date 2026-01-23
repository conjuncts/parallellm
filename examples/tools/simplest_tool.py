import logging

from pydantic import BaseModel
from parallellm.core.gateway import ParalleLLM
from parallellm.types import FunctionCallOutput
from dotenv import load_dotenv

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


def ls_tool(directory) -> str:
    return f"There are 4 files in {directory}."


with ParalleLLM.resume_directory(
    ".pllm/simplest-tool",
    provider="openai",
    strategy="sync",
    log_level=logging.DEBUG,
    # ignore_cache=True,
) as pllm:
    with pllm.agent(dashboard=True) as dash:
        # Structured output
        # resp = dash.ask_llm(
        # "Please name a power of 3.", hash_by=["llm"], text_format=MyModel
        # )

        # Tools
        msgs = ["How many files are in '~/examples'? Give the final answer in words."]
        resp = dash.ask_llm(
            msgs,
            hash_by=["llm"],
            tools=tools,
            # llm="gpt-4o"
        )

        dash.print(resp.resolve())
        tool_calls = resp.resolve_function_calls(to_dict=False)
        for call in tool_calls:
            dash.print(
                f"Tool call: `{call.name}` with args {call.args} call_id {call.call_id}"
            )

        assert len(tool_calls) == 1
        assert tool_calls[0].name == "count_files"
        msgs.append(resp)

        computed_tool_output = FunctionCallOutput(
            name=tool_calls[0].name,
            content=ls_tool(tool_calls[0].args),
            call_id=tool_calls[0].call_id,
        )

        resp = dash.ask_llm(msgs + [computed_tool_output], hash_by=["llm"])
        dash.print(resp.resolve())
        # Should respond by putting it in:
        # - function_call_output (openai)
        # - tool_result (anthropic)
