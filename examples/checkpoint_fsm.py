import logging
import random
from parallellm.core.gateway import ParalleLLM
from dotenv import load_dotenv
# from parallellm.core.manager import BatchManager


print("Before")

load_dotenv()

pllm = ParalleLLM.resume_directory(
    ".pllm/fsm",
    provider="openai",  #
    strategy="sync",
    log_level=logging.DEBUG,
)

# When code is deterministic, LLM calls be can directly cached.

# However, some code will be non-deterministic (ie. API calls, random)
# or might take a really long time, leading to different outcomes.

# In such a case, ParalleLLM introduces "checkpoints"

with pllm.default():
    # Non-checkpoint controlled: always executed
    best_vegetable = pllm.ask_llm("What is the best vegetable?")


with pllm.checkpoint():
    # Non-deterministic step
    pllm.when_checkpoint("random_step")
    num_steps = random.randint(3, 5)

    # IMPORTANT: save to userdata for later retrieval
    pllm.save_userdata("random_step/num_steps", num_steps)

    pllm.goto_checkpoint("generate_recipe")


with pllm.dashboard() as d:
    # pllm.dashboard() can be either non-checkpoint or checkpoint
    # depending on whether when_checkpoint() is called
    pllm.when_checkpoint("generate_recipe")

    random_num_steps = pllm.load_userdata("random_step/num_steps")
    recipe = pllm.ask_llm(
        f"Generate a recipe with {random_num_steps} steps using {best_vegetable.resolve()}.",
    )
    d.print(recipe.resolve())

# Why use checkpoints? If we rerun the code without checkpoints, then a different
# random number could be generated, which interferes with caching and
# leading to a different recipe.

pllm.persist()
