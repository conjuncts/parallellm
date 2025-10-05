"""
Test checkpoint and recipe examples

These tests validate:
1. Checkpoint functionality with deterministic and non-deterministic code
2. Userdata persistence across checkpoints
3. Recipe generation with mocked responses
"""

import pytest
import random
from unittest.mock import patch
from parallellm.core.gateway import ParalleLLM
from parallellm.testing.simple_mock import mock_openai_calls


def test_checkpoint_recipe_mocked(temp_integration_dir):
    """Test checkpoint recipe with mocked responses"""
    responses = [
        "**Carrots** are the best vegetable!",
        """Here's a 4-step carrot recipe:

1. Wash and peel 2 pounds of fresh carrots
2. Cut carrots into 1-inch pieces  
3. Boil in salted water for 15 minutes until tender
4. Season with butter, salt, and fresh herbs

Enjoy your delicious carrots!""",
    ]

    pllm = ParalleLLM.resume_directory(
        temp_integration_dir / "recipe", provider="openai", strategy="sync"
    )

    mock_client = mock_openai_calls(pllm, responses=responses)

    # First agent - get best vegetable
    agent = pllm.agent()
    with agent:
        best_vegetable = agent.ask_llm(
            "What is the best vegetable? Enclose your answer in **double asterisks**."
        )
        pllm.save_userdata("best_vegetable", best_vegetable)

    # Simulate non-deterministic step with fixed seed for testing
    with agent:
        agent.when_checkpoint("random")
        with patch("random.randint", return_value=4):  # Mock random to be deterministic
            num_steps = random.randint(3, 5)

        pllm.save_userdata("random/num_steps", num_steps)
        agent.goto_checkpoint("generate_recipe")

    # Generate recipe
    with pllm.agent() as dash:
        dash.when_checkpoint("generate_recipe")

        random_num_steps = pllm.load_userdata("random/num_steps")
        best_vegetable = pllm.load_userdata("best_vegetable")

        recipe = dash.ask_llm(
            f"Generate a recipe with {random_num_steps} steps using {best_vegetable.resolve()}."
        )

        recipe_text = recipe.resolve()
        assert "carrot" in recipe_text.lower()
        assert "4" in recipe_text  # Should have 4 steps
        assert "wash" in recipe_text.lower()

    # Verify API calls
    assert len(mock_client.calls) == 2

    # Check first call (vegetable question)
    first_call_input = mock_client._get_input_text(mock_client.calls[0]["input"])
    assert "vegetable" in first_call_input.lower()

    # Check second call (recipe generation)
    second_call_input = mock_client._get_input_text(mock_client.calls[1]["input"])
    assert "recipe" in second_call_input.lower()
    assert "4 steps" in second_call_input.lower()

    pllm.persist()


def test_checkpoint_persistence(temp_integration_dir):
    """Test that checkpoints and userdata persist correctly"""
    recipe_dir = temp_integration_dir / "recipe-checkpoint-persist"

    # Run 1: Set up userdata and checkpoint
    pllm1 = ParalleLLM.resume_directory(recipe_dir, provider="openai", strategy="sync")

    mock_client1 = mock_openai_calls(pllm1, responses=["**Spinach** is great!"])

    agent1 = pllm1.agent()
    with agent1:
        vegetable_resp = agent1.ask_llm("Best vegetable?")
        pllm1.save_userdata("test_vegetable", vegetable_resp)

    with agent1:
        agent1.when_checkpoint("test_checkpoint")
        pllm1.save_userdata("checkpoint_data", "checkpoint_value")
        agent1.goto_checkpoint("after_checkpoint")

    pllm1.persist()

    # Run 2: Load from persistence
    pllm2 = ParalleLLM.resume_directory(recipe_dir, provider="openai", strategy="sync")

    mock_client2 = mock_openai_calls(pllm2, responses=["Should not be called"])

    # Should be able to load userdata without API calls
    saved_vegetable = pllm2.load_userdata("test_vegetable")
    saved_checkpoint_data = pllm2.load_userdata("checkpoint_data")

    assert "Spinach" in saved_vegetable.resolve()
    assert saved_checkpoint_data == "checkpoint_value"

    # Verify no new API calls (data was persisted)
    assert len(mock_client2.calls) == 0

    pllm2.persist()


def test_multiple_checkpoint_workflow(temp_integration_dir):
    """Test complex workflow with multiple checkpoints"""
    responses = [
        "**Broccoli** is the healthiest!",
        "Steaming is the best cooking method",
        """Perfect 3-step broccoli recipe:
1. Steam broccoli florets for 5 minutes
2. Add garlic and olive oil  
3. Season with lemon and salt""",
    ]

    pllm = ParalleLLM.resume_directory(
        temp_integration_dir / "recipe-multi-checkpoint",
        provider="openai",
        strategy="sync",
    )

    mock_client = mock_openai_calls(pllm, responses=responses)

    agent = pllm.agent()

    # Phase 1: Get vegetable
    with agent:
        vegetable = agent.ask_llm("What's the healthiest vegetable?")
        pllm.save_userdata("vegetable", vegetable)

    # Phase 2: Get cooking method
    with agent:
        agent.when_checkpoint("cooking_method")
        method = agent.ask_llm("Best way to cook vegetables?")
        pllm.save_userdata("method", method)
        agent.goto_checkpoint("final_recipe")

    # Phase 3: Generate final recipe
    with agent:
        agent.when_checkpoint("final_recipe")

        veg = pllm.load_userdata("vegetable")
        cook_method = pllm.load_userdata("method")

        recipe = agent.ask_llm(
            f"Create a 3-step recipe for {veg.resolve()} using {cook_method.resolve()}"
        )

        final_recipe = recipe.resolve()
        assert "broccoli" in final_recipe.lower()
        assert "steam" in final_recipe.lower()
        assert "3" in final_recipe

    assert len(mock_client.calls) == 3
    pllm.persist()


def test_checkpoint_with_conditional_logic(temp_integration_dir):
    """Test checkpoints with conditional branching"""
    pllm = ParalleLLM.resume_directory(
        temp_integration_dir / "recipe-conditional", provider="openai", strategy="sync"
    )

    # Mock different responses based on conditions
    mock_client = mock_openai_calls(pllm)
    mock_client.add_patterns(
        {
            "vegetarian": "Great vegetarian recipe with beans and vegetables",
            "meat": "Delicious meat-based recipe with chicken and spices",
        }
    )

    agent = pllm.agent()

    # Simulate user preference (could be from input, config, etc.)
    preference = "vegetarian"  # Could be "meat" for different path

    with agent:
        agent.when_checkpoint(f"recipe_{preference}")

        if preference == "vegetarian":
            recipe = agent.ask_llm("Create a vegetarian recipe")
        else:
            recipe = agent.ask_llm("Create a meat recipe")

        result = recipe.resolve()
        assert "vegetarian" in result.lower()
        assert "beans" in result.lower()

    assert len(mock_client.calls) == 1
    pllm.persist()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
