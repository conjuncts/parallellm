from abc import ABC
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    List,
    Literal,
    TypedDict,
    Optional,
    Union,
    Tuple,
)
import json

from PIL import Image

if TYPE_CHECKING:
    from parallellm.core.identity import LLMIdentity


class AgentMetadata(TypedDict):
    """Metadata for an agent"""

    pass


class WorkingMetadata(TypedDict):
    agents: dict[str, AgentMetadata]

    session_counter: int
    """
    Numeric ID of session. Increments on each new BatchManager creation.
    """


class CallMetadata(TypedDict):
    """Not required metadata for a call."""

    provider_type: Optional[str]
    """Specific provider type (ie. openai, anthropic, etc)"""
    tag: Optional[str]
    """An optional tag to associate with the call."""


class CallIdentifier(TypedDict):
    agent_name: str
    """Name of the agent (if any, otherwise default) making the call."""
    doc_hash: str
    seq_id: int
    session_id: int
    """Numeric ID of session. For tracking/metadata purposes only."""

    meta: Optional[CallMetadata]


def to_serial_id(call_id: CallIdentifier, *, add_sess=True) -> str:
    if not add_sess:
        return f"{call_id['agent_name']}:{call_id['seq_id']}"
    return f"{call_id['agent_name']}:{call_id['seq_id']}:{call_id['session_id']}"


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


BatchStatus = Literal["ready", "error"]


@dataclass(slots=True)
class BatchResult:
    status: BatchStatus

    raw_output: Optional[str]
    """The raw output from the provider (for debugging/logging purposes)."""

    parsed_responses: Optional[List["ParsedResponse"]]
    """List of parsed responses, if available."""


class FunctionCall:
    """Represents a single tool call to a user-defined function."""

    __slots__ = ("name", "call_id", "args", "arg_str")

    def __init__(
        self,
        name: str,
        arguments: Union[str, dict],
        call_id: str,
    ):
        """
        Initialize a FunctionCall.

        Args:
            name: The name of the function being called.
            call_id: The unique identifier for this function call.
            arguments: The arguments for the function call as a dictionary.
            arg_str: The arguments for the function call as a JSON string.
        """
        self.name = name
        self.call_id = call_id

        if isinstance(arguments, str):
            # Parse arg_str to arguments
            self.args = json.loads(arguments)
        else:
            self.args = arguments
        self.arg_str = json.dumps(self.args)

    def __iter__(self):
        """Allow unpacking into tuple for backward compatibility."""
        return iter((self.name, self.args, self.call_id))

    def __repr__(self):
        return f"FunctionCall(name={self.name}, call_id={(self.call_id or '')[:8]}, args={self.args})"

    def __str__(self):
        return self.__repr__()


@dataclass(slots=True)
class FunctionCallRequest:
    """Represents the LLM requesting function/tool call(s)"""

    text_content: str
    """Text content, like thoughts about invoking a function."""

    calls: List[FunctionCall]
    """List of function calls."""

    call_id: CallIdentifier

    def __repr__(self):
        brief_calls = [
            f"{call.name}({(call.call_id or '')[:8]})" for call in self.calls
        ]
        return f"FunctionCallRequest(text_content={self.text_content}, calls={brief_calls})"

    def __str__(self):
        return self.__repr__()


@dataclass(slots=True)
class FunctionCallOutput:
    """Represents the output/result of a function/tool call."""

    content: str
    """The output content from the function call."""

    call_id: str
    """The ID of the function call this output corresponds to."""

    name: str
    """The name of the function call this output corresponds to."""

    def __repr__(self):
        return f"FunctionCallOutput(name={self.name}, call_id={(self.call_id or '')[:8]}, content={self.content[:20]}...)"

    def __str__(self):
        return self.__repr__()


ServerToolType = Literal["web_search", "code_interpreter"]


class ServerTool(ABC):
    """
    Represents a tool implemented by the server, such as web search or code interpreter.
    """

    server_tool_type: ServerToolType
    """Must be set by subclasses to identify the provider type."""

    kwargs: dict
    """Extra custom kwargs"""


LLMDocument = Union[
    str,
    Image.Image,
    Tuple[Literal["user", "assistant", "system", "developer"], str],
    FunctionCallRequest,
    FunctionCallOutput,
]
"""
Type alias for documents that can be either text or images.
"""

DocumentType = Literal["text", "function_call", "function_call_output", "llm_response"]
"""
Enum for valid document types. Closely matches OpenAI's document types.
LLMResponse: Any response from the LLM.
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

    function_calls: Optional[List[FunctionCall]] = None

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
    text_format: Optional[dict]
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

    batch_wait_until_complete: bool = True
    "Whether to wait for all batches to complete before proceeding."


@dataclass(slots=True)
class ParsedError:
    """
    Represents a parsed error from an LLM provider.
    """

    msg: str
    """The error message."""

    err_code: int
    """The error code, such as 429, 500, or 503."""

    error_id: Optional[str]
    """A unique identifier for this error from the provider."""

    metadata: Optional[dict]
    """Additional metadata from the provider (usage stats, model info, etc.)."""
