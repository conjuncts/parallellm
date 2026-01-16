import logging
import os
from parallellm.core.gateway import ParalleLLM
from dotenv import load_dotenv

# load_dotenv()

# Mess up the API key
os.environ["GEMINI_API_KEY"] = "invalid_key"

with ParalleLLM.resume_directory(
    ".pllm/simple/error",
    provider="google",
    strategy="sync",
    log_level=logging.DEBUG,
) as pllm:
    with pllm.agent(dashboard=True) as dash:
        resp = dash.ask_llm("What is 3 cubed?", hash_by=["llm"])

        dash.print(resp.resolve())
