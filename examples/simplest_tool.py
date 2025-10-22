import logging

from pydantic import BaseModel
from parallellm.core.gateway import ParalleLLM
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
    # ".temp",
    provider="anthropic",  #
    strategy="sync",
    log_level=logging.DEBUG,
    user_confirmation=True,
    ignore_cache=True,
) as pllm:
    with pllm.agent(dashboard=True) as dash:
        # Structured output
        # resp = dash.ask_llm(
        # "Please name a power of 3.", hash_by=["llm"], text_format=MyModel
        # )

        # Tools
        msgs = ["How many files are in '~/examples'?"]
        resp = dash.ask_llm(
            msgs,
            hash_by=["llm"],
            tools=tools,
        )

        dash.print(resp.resolve())
        tool_calls = resp.resolve_tool_calls(to_dict=False)
        for name, args, call_id in tool_calls:
            dash.print(f"Tool call: `{name}` with args {args} call_id {call_id}")

        assert len(tool_calls) == 1
        assert tool_calls[0][0] == "count_files"

        # openai
        # https://platform.openai.com/docs/guides/function-calling#custom-tools
        # computed_tool_output = {
        #     "type": "function_call_output",
        #     "call_id": tool_calls[0][2],
        #     "output": ls_tool(tool_calls[0][1]),
        # }

        # anthropic
        # https://docs.claude.com/en/docs/agents-and-tools/tool-use/implement-tool-use
        # computed_tool_output = {
        #     "role": "user",
        #     "content": [
        #         {
        #             "type": "tool_result",
        #             "tool_use_id": tool_calls[0][2],
        #             "output": ls_tool(tool_calls[0][1]),
        #         },
        #         {
        #             "type": "text",
        #             "text": "What should I do next?",
        #         },  # ✅ Text after tool_result
        #     ],
        # }

        # resp = dash.ask_llm(msgs + [computed_tool_output], hash_by=["llm"])
        # dash.print(resp.resolve())
        # Should respond by putting it in:
        # - function_call_output (openai)
        # - tool_result (anthropic)

# pllm.persist()
