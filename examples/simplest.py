import logging
from parallellm.core.gateway import ParalleLLM
from dotenv import load_dotenv

load_dotenv()

with ParalleLLM.resume_directory(
    ".pllm/simplest",
    # ".temp",
    provider="openai",  #
    strategy="batch",
    log_level=logging.DEBUG,
    user_confirmation=True,
    # ignore_cache=True,
) as pllm:
    # with pllm.default():
    with pllm.agent(dashboard=True) as dash:
        resp = dash.ask_llm("Please name a power of 3.", hash_by=["llm"])

        dash.print(resp.resolve())


# pllm.persist()
