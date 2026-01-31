import logging
from parallellm.core.gateway import ParalleLLM
from dotenv import load_dotenv

from PIL import Image

load_dotenv()

with ParalleLLM.resume_directory(
    ".pllm/simplest",
    provider="google",
    strategy="sync",
    log_level=logging.DEBUG,
    # ignore_cache=True,
) as pllm:
    with pllm.agent(dashboard=True) as dash:
        img = Image.open("tests/data/images/Nokota_Horses_cropped.jpg")
        resp = dash.ask_llm("What animal is this?", img, hash_by=["llm"])

        dash.print(resp.resolve())
