import logging
import os
import shutil
from parallellm.core.gateway import ParalleLLM
from dotenv import load_dotenv

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
        "Please name 8 enzymes. Place your final answer in a code block, separated by newlines.",
    )

    teams = resp.resolve().split("```")[1].split("\n")[1:9]

    d.print(f"===Candidates===")
    d.print(teams)
    # d.print()

    while len(teams) > 1:
        print(f"===Round of {len(teams)}===")
        responses = []
        for i in range(0, len(teams), 2):
            resp = pllm.ask_llm(
                f"Given two enzymes, choose the one you like more. Only respond with the name of the enzyme.",
                [teams[i], teams[i + 1]],
            )
            # do NOT call resp.resolve() in the hot loop
            responses.append(resp)

        # generate new round of teams
        teams = []
        for resp in responses:
            teams.append(resp.resolve())

        d.print("Teams:", teams)
        # d.print()
    d.print("===Winner===")
    d.print(teams[0])
    d.print("===Tournament Complete===")

# Finalize hash display before final print statements

pllm.persist()
