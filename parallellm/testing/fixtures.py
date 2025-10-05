"""Simple fixtures for mocking OpenAI responses"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class MockResponse:
    """Represents a mock response from an LLM"""

    output_text: str
    model: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    response_id: Optional[str] = None

    def __post_init__(self):
        """Set default usage if not provided"""
        if self.usage is None:
            self.usage = {
                "prompt_tokens": 10,
                "completion_tokens": len(self.output_text.split()),
                "total_tokens": 10 + len(self.output_text.split()),
            }
        if self.response_id is None:
            self.response_id = "mock-response-id"


class MockResponseQueue:
    """Manages a queue of mock responses for sequential calls"""

    def __init__(self, responses: List[Union[str, MockResponse]]):
        self.responses = [
            MockResponse(output_text=resp) if isinstance(resp, str) else resp
            for resp in responses
        ]
        self.index = 0

    def get_next_response(self) -> MockResponse:
        """Get the next response in the queue"""
        if self.index >= len(self.responses):
            raise IndexError(
                f"No more mock responses available. Called {self.index + 1} times but only {len(self.responses)} responses provided."
            )

        response = self.responses[self.index]
        self.index += 1
        return response

    def reset(self):
        """Reset the response queue to the beginning"""
        self.index = 0

    @property
    def is_exhausted(self) -> bool:
        """Check if all responses have been consumed"""
        return self.index >= len(self.responses)

    @property
    def remaining_count(self) -> int:
        """Get the number of remaining responses"""
        return max(0, len(self.responses) - self.index)


class MockResponseMap:
    """Maps instructions or patterns to specific responses"""

    def __init__(self):
        self.exact_matches: Dict[str, MockResponse] = {}
        self.pattern_matches: List[tuple] = []  # (pattern, response)
        self.default_response: Optional[MockResponse] = None

    def add_exact_match(self, instructions: str, response: Union[str, MockResponse]):
        """Add an exact instruction match"""
        if isinstance(response, str):
            response = MockResponse(output_text=response)
        self.exact_matches[instructions] = response

    def add_pattern_match(self, pattern: str, response: Union[str, MockResponse]):
        """Add a pattern-based match (simple substring matching)"""
        if isinstance(response, str):
            response = MockResponse(output_text=response)
        self.pattern_matches.append((pattern, response))

    def set_default_response(self, response: Union[str, MockResponse]):
        """Set a default response for unmatched instructions"""
        if isinstance(response, str):
            response = MockResponse(output_text=response)
        self.default_response = response

    def get_response(self, instructions: str) -> MockResponse:
        """Get a response for the given instructions"""
        # Check exact matches first
        if instructions in self.exact_matches:
            return self.exact_matches[instructions]

        # Check pattern matches
        for pattern, response in self.pattern_matches:
            if pattern.lower() in instructions.lower():
                return response

        # Return default if available
        if self.default_response:
            return self.default_response

        # If no matches, create a generic response
        return MockResponse(output_text=f"Mock response for: {instructions[:50]}...")
