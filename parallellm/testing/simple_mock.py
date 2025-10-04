"""Simple testing utilities for mocking OpenAI calls in ParalleLLM"""

from dataclasses import asdict
from typing import List, Union, Optional, Dict, Any
from unittest.mock import Mock, patch
import asyncio

from parallellm.core.manager import AgentOrchestrator

from .fixtures import MockResponse, MockResponseQueue, MockResponseMap


class MockOpenAIClient:
    """Mock OpenAI client that records calls and returns predefined responses"""

    def __init__(self):
        self.calls = []
        self.response_queue: Optional[MockResponseQueue] = None
        self.response_map: Optional[MockResponseMap] = None
        self.responses = Mock()
        self.responses.create = self._create_response

    def set_responses(self, responses: List[Union[str, MockResponse]]):
        """Set a queue of responses to return sequentially"""
        self.response_queue = MockResponseQueue(responses)
        self.response_map = None

    def set_response_map(self, response_map: MockResponseMap):
        """Set a response map for pattern-based responses"""
        self.response_map = response_map
        self.response_queue = None

    def add_response_pattern(self, pattern: str, response: Union[str, MockResponse]):
        """Add a single response pattern"""
        if self.response_map is None:
            self.response_map = MockResponseMap()
            self.response_queue = None
        self.response_map.add_pattern_match(pattern, response)

    def set_default_response(self, response: Union[str, MockResponse]):
        """Set a default response"""
        if self.response_map is None:
            self.response_map = MockResponseMap()
            self.response_queue = None
        self.response_map.set_default_response(response)

    def _get_mock_response(self, input_messages: List) -> MockResponse:
        """Get the appropriate mock response based on input messages"""
        if self.response_queue is not None:
            return self.response_queue.get_next_response()
        elif self.response_map is not None:
            # Extract text from input messages for pattern matching
            combined_text = self._extract_text_from_input(input_messages)
            return self.response_map.get_response(combined_text)
        else:
            combined_text = self._extract_text_from_input(input_messages)
            return MockResponse(
                output_text=f"Mock response for: {combined_text[:50]}..."
            )

    def _extract_text_from_input(self, input_messages: List) -> str:
        """Extract text content from input messages for pattern matching"""
        if not input_messages:
            return ""

        # Handle both string messages and dict-like message objects
        text_parts = []
        for msg in input_messages:
            if isinstance(msg, str):
                text_parts.append(msg)
            elif isinstance(msg, dict) and "content" in msg:
                text_parts.append(msg["content"])
            elif hasattr(msg, "content"):  # For message objects
                text_parts.append(msg.content)
            else:
                # Fallback: convert to string
                text_parts.append(str(msg))

        return " ".join(text_parts)

    def _create_response(self, model=None, instructions=None, input=None, **kwargs):
        """Mock the responses.create method"""
        # Record the call
        call = {
            "model": model,
            "instructions": instructions,
            "input": input or [],
            "kwargs": kwargs,
        }
        self.calls.append(call)

        # Get mock response based on input messages
        mock_response = self._get_mock_response(input or [])

        # Need to return the full response (as dict)
        return asdict(mock_response)

    def clear_calls(self):
        """Clear all recorded calls"""
        self.calls.clear()

    @property
    def call_count(self) -> int:
        """Get the number of calls made"""
        return len(self.calls)

    def get_last_call(self) -> Optional[Dict[str, Any]]:
        """Get the most recent call"""
        return self.calls[-1] if self.calls else None


class MockAsyncOpenAIClient(MockOpenAIClient):
    """Async version of mock OpenAI client"""

    def __init__(self):
        super().__init__()
        self.responses.create = self._async_create_response

    async def _async_create_response(
        self, model=None, instructions=None, input=None, **kwargs
    ):
        """Mock the async responses.create method"""
        # Record the call
        call = {
            "model": model,
            "instructions": instructions,
            "input": input or [],
            "kwargs": kwargs,
        }
        self.calls.append(call)

        # Get mock response based on input messages
        mock_response = self._get_mock_response(input or [])

        # Simulate async delay
        await asyncio.sleep(0.01)

        return mock_response


def mock_openai_calls(
    batch_manager: AgentOrchestrator,
    responses: Optional[List[Union[str, MockResponse]]] = None,
    response_map: Optional[MockResponseMap] = None,
) -> Union[MockOpenAIClient, MockAsyncOpenAIClient]:
    """
    Replace the OpenAI client in a BatchManager with a mock client

    Args:
        batch_manager: The BatchManager instance to modify
        responses: List of responses to return sequentially
        response_map: Map of instruction patterns to responses

    Returns:
        The mock client that was installed (for call inspection)
    """
    provider = batch_manager._provider

    # Determine if we need sync or async client based on provider type
    from parallellm.provider.openai import SyncOpenAIProvider, AsyncOpenAIProvider

    if isinstance(provider, SyncOpenAIProvider):
        mock_client = MockOpenAIClient()
    elif isinstance(provider, AsyncOpenAIProvider):
        mock_client = MockAsyncOpenAIClient()
    else:
        # Default to sync
        mock_client = MockOpenAIClient()

    # Set up responses
    if responses is not None:
        mock_client.set_responses(responses)
    elif response_map is not None:
        mock_client.set_response_map(response_map)

    # Replace the client in the provider
    provider.client = mock_client

    return mock_client


# Convenience functions
def create_mock_responses(responses: List[str]) -> List[MockResponse]:
    """Convert a list of strings to MockResponse objects"""
    return [MockResponse(output_text=resp) for resp in responses]


def create_response_map(**patterns) -> MockResponseMap:
    """Create a response map from keyword arguments"""
    response_map = MockResponseMap()
    for pattern, response in patterns.items():
        response_map.add_pattern_match(pattern, response)
    return response_map


# Simple assertion helpers
def assert_call_made(mock_client, content_substring: str):
    """Assert that a call containing the given substring in input was made"""
    for call in mock_client.calls:
        # Check both instructions and input content
        instructions = call.get("instructions", "") or ""
        input_messages = call.get("input", [])

        # Extract text from input messages
        input_text = mock_client._extract_text_from_input(input_messages)
        combined_text = f"{instructions} {input_text}".strip()

        if content_substring.lower() in combined_text.lower():
            return True

    raise AssertionError(f"No call found containing '{content_substring}'")


def assert_call_count(mock_client, expected_count: int):
    """Assert the number of calls made to the mock client"""
    actual_count = mock_client.call_count
    if actual_count != expected_count:
        raise AssertionError(f"Expected {expected_count} calls, but got {actual_count}")


def get_call_content(mock_client) -> List[str]:
    """Get all call content (both instructions and input) for inspection"""
    content_list = []
    for call in mock_client.calls:
        instructions = call.get("instructions", "") or ""
        input_messages = call.get("input", [])
        input_text = mock_client._extract_text_from_input(input_messages)
        combined_text = f"{instructions} {input_text}".strip()
        content_list.append(combined_text)
    return content_list


def get_call_instructions(mock_client) -> List[str]:
    """Get all call instructions for inspection (legacy function)"""
    return [call.get("instructions", "") for call in mock_client.calls]
