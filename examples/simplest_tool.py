import logging

from pydantic import BaseModel
from parallellm.core.gateway import ParalleLLM
from dotenv import load_dotenv

load_dotenv()


class MyModel(BaseModel):
    final_answer: str


with ParalleLLM.resume_directory(
    ".pllm/simplest-tool",
    # ".temp",
    provider="openai",  #
    strategy="sync",
    log_level=logging.DEBUG,
    user_confirmation=True,
    # ignore_cache=True,
) as pllm:
    # with pllm.default():
    with pllm.agent(dashboard=True) as dash:
        resp = dash.ask_llm(
            "Please name a power of 3.", hash_by=["llm"], text_format=MyModel
        )

        dash.print(resp.resolve())


# pllm.persist()
