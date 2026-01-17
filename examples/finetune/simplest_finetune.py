import logging
from parallellm.core.gateway import ParalleLLM
from dotenv import load_dotenv

load_dotenv()

with ParalleLLM.resume_directory(
    ".pllm/simple/finetune",
    provider="google",
    strategy="sync",
    log_level=logging.DEBUG,
    # ignore_cache=True,
) as pllm:
    # with pllm.default():
    with pllm.agent(dashboard=True) as dash:
        resp = dash.ask_llm(
            "Please name a power of 3.", hash_by=["llm"], tag="power-of-3"
        )

        dash.print(resp.resolve())
