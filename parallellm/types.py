from dataclasses import dataclass
from typing import List, Literal, TypedDict, Optional, Union, Tuple

from PIL import Image


class AgentMetadata(TypedDict):
    checkpoint_counter: int
    """
    Counter for checkpoint mode operations. Persisted across runs.
    """

    latest_checkpoint: Optional[str]
    """
    Latest checkpoint is that which has last been reached
    """


class WorkingMetadata(TypedDict):
    agents: dict[str, AgentMetadata]

    session_counter: int
    """
    Numeric ID of session. Increments on each new BatchManager creation.
    """


class CallIdentifier(TypedDict):
    agent_name: str
    """Name of the agent (if any, otherwise default) making the call."""
    checkpoint: Optional[str]
    doc_hash: str
    seq_id: int
    session_id: int
    """Numeric ID of session. For tracking/metadata purposes only."""

    provider_type: Optional[str]
    """Specific provider type (ie. openai, anthropic, etc)"""


@dataclass
class BatchIdentifier:
    call_ids: List[CallIdentifier]

    custom_ids: List[str]
    """Custom IDs assigned to each call in the batch, in the same order as call_ids."""

    batch_uuid: str
    """A unique identifier given for the batch by the provider."""


@dataclass
class CohortIdentifier:
    """
    If a batch is very large (>1000), it may have to be split into multiple sub-batches
    to keep below file size limits.

    A cohort is a collection of such sub-batches that together represent a single logical batch.
    """

    batch_ids: List[BatchIdentifier]

    session_id: int
    """Numeric ID of session. Here, it also serves as a cohort ID."""


class AskParameters(TypedDict):
    """Parameters for ask_llm()."""

    hash_by: Optional[List[Literal["llm"]]]


BatchStatus = Literal["pending", "ready", "error"]


@dataclass
class BatchResult:
    status: BatchStatus

    raw_output: Optional[str]
    """The raw output from the provider (for debugging/logging purposes)."""

    parsed_responses: Optional[List["ParsedResponse"]]
    """List of parsed responses, if available."""


# Type alias for documents that can be either text or images
LLMDocument = Union[
    str, Image.Image, Tuple[Literal["user", "assistant", "system", "developer"], str]
]


ProviderType = Literal["openai", "anthropic", "google"]


@dataclass(slots=True)
class ParsedResponse:
    """
    Represents a parsed response from an LLM provider.

    This dataclass encapsulates the three key components of a parsed response:
    - The response text content
    - The response ID (if available from the provider)
    - Additional metadata from the provider
    """

    text: str
    """The main text content of the response."""

    response_id: Optional[str]
    """The unique identifier for this response from the provider."""

    metadata: Optional[dict]
    """Additional metadata from the provider (usage stats, model info, etc.)."""

    tool_calls: Optional[List[Tuple[str, str, str]]] = None

    def __iter__(self):
        """Allow unpacking into tuple for backward compatibility."""
        return iter((self.text, self.response_id, self.metadata))
