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
    agent_name: Optional[str]
    """Name of the agent (if any, otherwise default) making the call."""
    checkpoint: Optional[str]
    doc_hash: str
    seq_id: int
    session_id: int
    """Numeric ID of session. For tracking/metadata purposes only."""

    provider_type: Optional[str]
    """Specific provider type (ie. openai, anthropic, etc)"""


class AskParameters(TypedDict):
    """Parameters for ask_llm()."""

    salt_by: Optional[List[Literal["llm"]]]


# Type alias for documents that can be either text or images
LLMDocument = Union[str, Image.Image]
