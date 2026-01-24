import logging

from pydantic import BaseModel
from parallellm.core.gateway import ParalleLLM
from parallellm.types import FunctionCallOutput
from dotenv import load_dotenv

load_dotenv()


class MyModel(BaseModel):
    final_answer: str


with ParalleLLM.resume_directory(
    ".pllm/simplest-tool",
    provider="google",
    strategy="batch",
    log_level=logging.DEBUG,
    # ignore_cache=True,
) as pllm:
    with pllm.agent(dashboard=True) as dash:
        # Structured output
        resp = dash.ask_llm(
            "Please name a power of 3.", hash_by=["llm"], text_format=MyModel
        )

        dash.print(resp.resolve())
