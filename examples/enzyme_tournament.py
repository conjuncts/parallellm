import logging
import os
import shutil
import time
from parallellm.core.gateway import ParalleLLM
from dotenv import load_dotenv

from parallellm.core.response import LLMResponse

start = time.time()
load_dotenv()

shutil.rmtree(".temp", ignore_errors=True)
pllm = ParalleLLM.resume_directory(
    # ".pllm",
    ".temp",
    provider="openai",  #
    strategy="async",
    log_level=logging.DEBUG,
)

# with pllm.default():
with pllm.dashboard() as d:
    d.print("===Starting Tournament===")
    resp = pllm.ask_llm(
        "You are a helpful assistant",
        "Please name 32 enzymes. Place your final answer in a code block, separated by newlines.",
    )

    teams = [x for x in resp.resolve().split("```")[1].split("\n")[1:] if x]

    d.print(f"===Candidates===")
    d.print(teams)

    while len(teams) > 1:
        print(f"===Round of {len(teams)}===")
        responses = []
        for i in range(0, len(teams), 2):
            if i + 1 < len(teams):
                resp = pllm.ask_llm(
                    f"Given two enzymes, choose the one you like more. Only respond with the name of the enzyme.",
                    [teams[i], teams[i + 1]],
                )
                # do NOT call resp.resolve() in the hot loop
                responses.append(resp)
            else:
                # they win by default
                responses.append(LLMResponse(teams[i]))

        # Resolve only once everything is submitted
        teams = []
        for resp in responses:
            teams.append(resp.resolve())

        d.print("Teams:", teams)
    d.print("===Winner===")
    d.print(teams[0])

# Finalize hash display before final print statements

pllm.persist()
print("Total time:", time.time() - start)
# 32 enzymes, sync: 23.34247851371765s
# 32 enzymes: async: 7.341606140136719
# cached: 0.06931805610656738s
