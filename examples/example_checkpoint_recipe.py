import logging
import random
from parallellm.core.gateway import ParalleLLM
from dotenv import load_dotenv
# from parallellm.core.manager import BatchManager

load_dotenv()

pllm = ParalleLLM.resume_directory(
    ".pllm/recipe",
    provider="openai",
    strategy="sync",
    log_level=logging.DEBUG,
)

# When code is deterministic, LLM calls be can directly cached.

# However, some code will be non-deterministic (ie. API calls, random)
# or might take a really long time, leading to different outcomes.

# In such a case, ParalleLLM introduces "checkpoints"
agent = pllm.agent()

with agent:
    # Non-checkpoint controlled: always executed
    best_vegetable = agent.ask_llm(
        "What is the best vegetable? Enclose your answer in **double asterisks**."
    )
    pllm.save_userdata("best_vegetable", best_vegetable)  # save for later


with agent:
    # Non-deterministic step
    agent.when_checkpoint("random")
    num_steps = random.randint(3, 5)

    # IMPORTANT: save to userdata for later retrieval
    pllm.save_userdata("random/num_steps", num_steps)
    agent.goto_checkpoint("generate_recipe")


with pllm.agent(dashboard=True) as dash:
    # pllm.dashboard() can be either non-checkpoint or checkpoint
    # depending on whether when_checkpoint() is called
    dash.when_checkpoint("generate_recipe")

    random_num_steps = pllm.load_userdata("random/num_steps")
    best_vegetable = pllm.load_userdata("best_vegetable")
    recipe = dash.ask_llm(
        f"Generate a recipe with {random_num_steps} steps using {best_vegetable.resolve()}.",
    )
    dash.print(recipe.resolve())

# Why use checkpoints? If we rerun the code without checkpoints, then a different
# random number could be generated, which interferes with caching and
# leading to a different recipe.

pllm.persist()
