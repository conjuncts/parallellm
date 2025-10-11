import logging
from parallellm.core.gateway import ParalleLLM
from dotenv import load_dotenv

load_dotenv()

pllm = ParalleLLM.resume_directory(
    ".pllm/simplest",
    # ".temp",
    provider="anthropic",  #
    strategy="sync",
    log_level=logging.DEBUG,
    # ignore_cache=True,
)

# with pllm.default():
with pllm.agent(dashboard=True) as dash:
    resp = dash.ask_llm("Please name a power of 2.", hash_by=["llm"])

    dash.print(resp.resolve())


pllm.persist()
