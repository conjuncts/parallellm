from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal, TypedDict, Optional, Union, Tuple
import json

from PIL import Image

if TYPE_CHECKING:
    from parallellm.core.identity import LLMIdentity


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


@dataclass(slots=True)
class BatchResult:
    status: BatchStatus

    raw_output: Optional[str]
    """The raw output from the provider (for debugging/logging purposes)."""

    parsed_responses: Optional[List["ParsedResponse"]]
    """List of parsed responses, if available."""


class ToolCall:
    """Represents a single tool/function call."""

    __slots__ = ("name", "call_id", "args", "arg_str")

    def __init__(
        self,
        name: str,
        arguments: Union[str, dict],
        call_id: str,
    ):
        """
        Initialize a ToolCall.

        Args:
            name: The name of the tool/function being called.
            call_id: The unique identifier for this tool call.
            arguments: The arguments for the tool call as a dictionary.
            arg_str: The arguments for the tool call as a JSON string.
        """
        self.name = name
        self.call_id = call_id

        if isinstance(arguments, str):
            # Parse arg_str to arguments
            self.args = json.loads(arguments)
            self.arg_str = arguments
        else:
            self.args = arguments
            self.arg_str = json.dumps(arguments)

    def __iter__(self):
        """Allow unpacking into tuple for backward compatibility."""
        return iter((self.name, self.args, self.call_id))


@dataclass(slots=True)
class ToolCallRequest:
    """Represents the LLM requesting function/tool call(s)"""

    text_content: str
    """Text content, like thoughts about invoking a tool."""

    calls: List[ToolCall]
    """List of tool calls."""

    def __iter__(self):
        """Allow unpacking into tuple for backward compatibility."""
        return iter(("function_call", self.calls))


@dataclass(slots=True)
class ToolCallOutput:
    """Represents the output/result of a function/tool call."""

    content: str
    """The output content from the function call."""

    call_id: str
    """The ID of the function call this output corresponds to."""

    name: str
    """The name of the function call this output corresponds to."""

    def __iter__(self):
        """Allow unpacking into tuple for backward compatibility."""
        return iter(("function_call_output", (self.content, self.call_id)))


LLMDocument = Union[
    str,
    Image.Image,
    Tuple[Literal["user", "assistant", "system", "developer"], str],
    ToolCallRequest,
    ToolCallOutput,
]
"""
Type alias for documents that can be either text or images.
"""


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

    tool_calls: Optional[List[ToolCall]] = None

    def __iter__(self):
        """Allow unpacking into tuple for backward compatibility."""
        return iter((self.text, self.response_id, self.metadata))


class CommonQueryParameters(TypedDict):
    """
    Common parameters for LLM calls across providers.
    """

    instructions: Optional[str]
    documents: Union[LLMDocument, List[LLMDocument]]
    llm: "LLMIdentity"
    text_format: Optional[str]
    tools: Optional[List[dict]]


@dataclass(frozen=True, slots=True)
class MinorTweaks:
    """
    Minor tweaks for the ParallelLLM framework.
    Holds configs not significant enough to warrant a full keyword argument.
    """

    async_max_concurrent: Optional[int] = 20
    "Maximum number of concurrent tasks in AsyncBackend."

    batch_user_confirmation: bool = True
    "Whether to ask for user confirmation before submitting a batch."
