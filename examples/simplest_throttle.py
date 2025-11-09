import logging
import time
from parallellm.core.gateway import ParalleLLM
from dotenv import load_dotenv

from parallellm.core.throttler import Throttler
from parallellm.types import MinorTweaks

load_dotenv()

with ParalleLLM.resume_directory(
    ".pllm/simplest-throttle",
    # ".temp",
    provider="openai",  #
    strategy="async",
    log_level=logging.DEBUG,
    ignore_cache=True,
    throttler=Throttler(
        max_requests_per_window=4,
        window_seconds=10,
    ),
    tweaks=MinorTweaks(
        async_max_concurrent=2,
    ),
) as pllm:
    # with pllm.default():
    time_start = time.time()
    with pllm.agent(dashboard=True) as dash:
        for i in range(5):
            req_start = time.time()
            resp = dash.ask_llm(f"Please name a power of {i + 2}.", hash_by=["llm"])
            # dash.print(resp.resolve())
            req_end = time.time()
            dash.print(
                f"Response {i} at {req_start - time_start} took {req_end - req_start:.2f}s"
            )


# pllm.persist()
