from typing import TYPE_CHECKING, List, Optional, Union
from pydantic import BaseModel
from parallellm.core.identity import LLMIdentity
from parallellm.provider.base import AsyncProvider, BaseProvider, SyncProvider
from parallellm.types import ParsedResponse, CommonQueryParameters
from parallellm.types import CallIdentifier, LLMDocument

if TYPE_CHECKING:
    from anthropic import Anthropic, AsyncAnthropic
    from anthropic.types import Message


def _fix_docs_for_anthropic(
    documents: Union[LLMDocument, List[LLMDocument]],
) -> List[dict]:
    """Ensure documents are in the correct format for Anthropic API"""
    if not isinstance(documents, list):
        documents = [documents]

    formatted_docs = []
    for doc in documents:
        if isinstance(doc, str):
            msg = {
                "role": "user",
                "content": doc,
            }
            formatted_docs.append(msg)
        elif isinstance(doc, tuple) and len(doc) == 2:
            # Handle Tuple[Literal["user", "assistant", "system", "developer"], str]
            role, content = doc
            msg = {
                "role": role,
                "content": content,
            }
            formatted_docs.append(msg)
        elif isinstance(doc, dict):
            # If it's already a proper message dict, keep it
            if "role" in doc and "content" in doc:
                formatted_docs.append(doc)
            elif "content" in doc:
                formatted_docs.append({"role": "user", "content": doc["content"]})
            elif "text" in doc:
                formatted_docs.append({"role": "user", "content": doc["text"]})
            else:
                formatted_docs.append({"role": "user", "content": str(doc)})
        else:
            formatted_docs.append({"role": "user", "content": str(doc)})

    return formatted_docs


def _prepare_anthropic_config(params: CommonQueryParameters, **kwargs) -> tuple:
    """Prepare config and messages for Anthropic API calls"""
    instructions = params["instructions"]
    documents = params["documents"]
    llm = params["llm"]
    tools = params.get("tools")

    if tools:
        tools = _prepare_tool_schema(tools)

    messages = _fix_docs_for_anthropic(documents)

    config = kwargs.copy()
    if instructions:
        config["system"] = instructions

    model_name = llm.model_name

    return model_name, messages, tools, config


def _prepare_tool_schema(func_schemas: List[dict]) -> List[dict]:
    """Convert tool definitions to Anthropic tool schema"""

    anthropic_tools = []
    for sch in func_schemas:
        sch2 = None
        if "type" in sch:
            # remove type field (used by openai)
            sch2 = sch.copy()
            sch2.pop("type")

        if "parameters" in sch:
            sch2 = sch2 or sch.copy()
            # rename parameters to input_schema
            sch2["input_schema"] = sch2.pop("parameters")

        if sch2 is not None:
            sch = sch2
        anthropic_tools.append(sch)
    return anthropic_tools


class AnthropicProvider(BaseProvider):
    provider_type: str = "anthropic"

    def get_default_llm_identity(self) -> LLMIdentity:
        return LLMIdentity("claude-3-haiku-20240307", provider=self.provider_type)

    def parse_response(self, raw_response: Union[BaseModel, dict]) -> ParsedResponse:
        """Parse Anthropic API response into common format"""

        # https://docs.claude.com/en/docs/agents-and-tools/tool-use/implement-tool-use
        if isinstance(raw_response, BaseModel):
            # Pydantic model (e.g., anthropic.types.Message)
            response: Message = raw_response
            text_contents = []
            tool_calls = []
            for content_item in response.content:
                if content_item.type == "text":
                    text_contents.append(content_item.text)
                elif content_item.type == "tool_use":
                    tool_calls.append(
                        (content_item.name, content_item.input, content_item.id)
                    )
            text_content = "".join(text_contents)

            resp_id = response.id
            obj = response.model_dump(mode="json")
            obj.pop("id", None)
            parsed_metadata = obj

        elif isinstance(raw_response, dict):
            # Dict response
            resp_id = raw_response.get("id", None)

            # Extract text and tool calls from content
            content = raw_response.get("content", [])
            text_contents = []
            tool_calls = []

            if isinstance(content, list):
                for content_item in content:
                    if not isinstance(content_item, dict):
                        text_contents.append(str(content_item))
                        continue

                    if content_item.get("type") == "text":
                        text_contents.append(content_item["text"])
                    elif content_item.get("type") == "tool_use":
                        tool_calls.append(
                            (
                                content_item.get("name"),
                                content_item.get("input"),
                                content_item.get("id"),
                            )
                        )
            else:
                text_contents.append(str(content))

            text_content = "".join(text_contents)

            # Create a copy for metadata to avoid mutating the original
            parsed_metadata = raw_response.copy()
            parsed_metadata.pop("id", None)
        else:
            raise ValueError(f"Unsupported response type: {type(raw_response)}")
        return ParsedResponse(
            text=text_content,
            response_id=resp_id,
            metadata=parsed_metadata,
            tool_calls=tool_calls,
        )


class SyncAnthropicProvider(SyncProvider, AnthropicProvider):
    def __init__(self, client: "Anthropic"):
        self.client = client

    def prepare_sync_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare a synchronous callable for Anthropic API"""
        model_name, messages, tools, config = _prepare_anthropic_config(
            params, **kwargs
        )

        return self.client.messages.create(
            model=model_name,
            max_tokens=config.pop("max_tokens", 4096),
            messages=messages,
            tools=tools,
            **config,
        )


class AsyncAnthropicProvider(AsyncProvider, AnthropicProvider):
    def __init__(self, client: "AsyncAnthropic"):
        self.client = client

    def prepare_async_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare an async coroutine for Anthropic API"""
        model_name, messages, tools, config = _prepare_anthropic_config(
            params, **kwargs
        )

        coro = self.client.messages.create(
            model=model_name,
            max_tokens=config.pop("max_tokens", 1024),
            messages=messages,
            tools=tools,
            **config,
        )

        return coro
