import logging
from parallellm.core.gateway import ParalleLLM
from dotenv import load_dotenv
# from parallellm.core.manager import BatchManager


print("Before")

load_dotenv()
pllm = ParalleLLM.resume_directory(".pllm", provider="openai", log_level=logging.DEBUG)

with pllm:
    print("This will always be executed")
    pllm.goto_stage("begin")

with pllm:
    pllm.when_stage("begin")

    resp = pllm.ask_llm(
        "You are a helpful assistant",
        "Please name 8 NFL teams. Place your final answer in a code block, separated by newlines.",
    )

    print(resp.resolve())

    teams = resp.resolve().split("```")[1].split("\n")[1:9]

    for i in range(0, len(teams), 2):
        resp = pllm.ask_llm(
            "You are a sports analyst",
            f"Predict the winner of the game between the {teams[i]} and the {teams[i + 1]}. Give a paragraph of reasoning.",
        )
        print(resp.resolve()[:70] + "...")

    pllm.goto_stage("end")

with pllm:
    pllm.when_stage("end")
print("After")

pllm.persist()
