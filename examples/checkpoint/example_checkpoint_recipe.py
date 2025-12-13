import logging
import random
from parallellm.core.gateway import ParalleLLM
from dotenv import load_dotenv

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
agent = pllm.agent(dashboard=True)

with agent:
    # Non-checkpoint controlled: always executed
    best_vegetable = agent.ask_llm(
        "What is the best vegetable? Enclose your final answer in **double asterisks**."
    )
    pllm.save_userdata("best_vegetable", best_vegetable)  # save for later


with agent:
    # Non-deterministic step
    agent.when_checkpoint("random")
    num_steps = random.randint(3, 5)

    # IMPORTANT: save to userdata for later retrieval
    pllm.save_userdata("random/num_steps", num_steps)
    agent.goto_checkpoint("generate_recipe")


with agent:
    # pllm.dashboard() can be either non-checkpoint or checkpoint
    # depending on whether when_checkpoint() is called
    agent.when_checkpoint("generate_recipe")

    random_num_steps = pllm.load_userdata("random/num_steps")
    previous_query = pllm.load_userdata("best_vegetable")
    best_vegetable = previous_query.resolve().split("**")[1]
    recipe = agent.ask_llm(
        f"Generate a recipe with {random_num_steps} steps using {best_vegetable}.",
    )
    agent.print(recipe.resolve())

# Why use checkpoints? If we rerun the code without checkpoints, then a different
# random number could be generated, which interferes with caching and
# leading to a different recipe.

pllm.persist()
