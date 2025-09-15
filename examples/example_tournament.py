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
    ".pllm",
    # ".temp",
    provider="openai",  #
    strategy="async",
    log_level=logging.DEBUG,
)

with pllm:
    print("This will always be executed")
    pllm.goto_stage("begin")

with pllm:
    pllm.when_stage("begin")

    with pllm.dashboard():
        resp = pllm.ask_llm(
            "You are a helpful assistant",
            "Please name 8 NFL teams. Place your final answer in a code block, separated by newlines.",
        )

        teams = resp.resolve().split("```")[1].split("\n")[1:9]

    print(f"Got teams: {teams}")

    with pllm.dashboard():
        games = []
        for i in range(0, len(teams), 2):
            resp = pllm.ask_llm(
                "You are a sports analyst",
                # f"Predict the winner of the game between the {teams[i]} and the {teams[i + 1]}. Give a paragraph of reasoning.",
                f"Given a game between the {teams[i]} and the {teams[i + 1]}, simply predict the winner and the score.",
            )
            # do NOT call resp.resolve() in the hot loop
            games.append(resp)

        # waiting for all to be submitted results in better batching!
        game_descriptions = []
        for resp in games:
            game_descriptions.append(resp.resolve())

    print("Descriptions:", [x[:70] for x in game_descriptions])
    # Finalize hash logger display before stage change
    pllm.goto_stage("end")

with pllm:
    pllm.when_stage("end")
    print("Inside stage 'end'")

# Finalize hash display before final print statements
print("After")

pllm.persist()
