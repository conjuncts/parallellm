from dataclasses import dataclass
from typing import List, Literal, TypedDict, Optional, Union

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


# Type alias for documents that can be either text or images
LLMDocument = Union[str, Image.Image]


ProviderType = Literal["openai", "anthropic", "google"]
