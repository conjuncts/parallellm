import logging
import os
import shutil
from parallellm.core.gateway import ParalleLLM
from dotenv import load_dotenv
# from parallellm.core.manager import BatchManager


print("Before")

load_dotenv()

shutil.rmtree(".temp", ignore_errors=True)
pllm = ParalleLLM.resume_directory(
    # ".pllm",
    ".temp",
    provider="openai",  #
    strategy="sync",
    log_level=logging.DEBUG,
)

# with pllm.default():
with pllm.dashboard() as d:
    d.print("This will always be executed")

    resp = pllm.ask_llm(
        "Please name 8 NFL teams. Place your final answer in a code block, separated by newlines.",
    )

    teams = resp.resolve().split("```")[1].split("\n")[1:9]

    d.print(f"Got teams: {teams}")

    games = []
    for i in range(0, len(teams), 2):
        resp = pllm.ask_llm(
            f"Given a game between the {teams[i]} and the {teams[i + 1]}, simply predict the winner and the score.",
        )
        d.print("Asked!")
        # do NOT call resp.resolve() in the hot loop
        games.append(resp)

    # waiting for all to be submitted results in better batching!
    game_descriptions = []
    for resp in games:
        game_descriptions.append(resp.resolve())

    d.print("Descriptions:", [x[:70] for x in game_descriptions])
    # Finalize hash logger display before stage change

with pllm.default():
    pllm.when_stage("begin")  # stage-controlled - needs to modify the global counter
    print("Inside stage 'begin'")
    pllm.goto_stage("end")

with pllm.default():
    pllm.when_stage("end")
    print("Inside stage 'end'")

# Finalize hash display before final print statements
print("After")

pllm.persist()
