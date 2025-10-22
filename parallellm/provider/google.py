import json
from typing import TYPE_CHECKING, List, Optional, Union
from pydantic import BaseModel
from parallellm.core.identity import LLMIdentity
from parallellm.provider.base import AsyncProvider, BaseProvider, SyncProvider
from parallellm.types import ParsedResponse, CommonQueryParameters
from parallellm.types import CallIdentifier, LLMDocument

from google import genai
from google.genai import types


def _fix_docs_for_google(
    documents: Union[LLMDocument, List[LLMDocument]],
) -> Union[str, List]:
    """Ensure documents are in the correct format for Gemini API"""
    if not isinstance(documents, list):
        documents = [documents]

    # For Gemini, we can pass strings directly or convert to proper format
    # The SDK will handle the conversion automatically
    if len(documents) == 1 and isinstance(documents[0], str):
        return documents[0]  # Single string can be passed directly

    # For multiple documents or mixed types, return as list
    formatted_docs = []
    for doc in documents:
        if isinstance(doc, str):
            formatted_docs.append(doc)
        elif isinstance(doc, tuple) and len(doc) == 2:
            # Handle Tuple[Literal["user", "assistant", "system", "developer"], str]
            role, content = doc
            # For Google, we might need to format this differently
            # For now, just append the content with a prefix
            if role == "assistant":
                role = "model"
            elif role != "user":
                raise ValueError(f"Unsupported role for Google: {role}")
            formatted_docs.append(
                {
                    "role": role,
                    "parts": [{"text": content}],
                }
            )
        elif isinstance(doc, dict):
            # If it's already a proper content dict, keep it
            if "parts" in doc and "role" in doc:
                formatted_docs.append(doc)
            elif "content" in doc:
                formatted_docs.append(doc["content"])
            elif "text" in doc:
                formatted_docs.append(doc["text"])
            else:
                formatted_docs.append(str(doc))
        else:
            formatted_docs.append(str(doc))

    return formatted_docs


def _prepare_tool_schema(func_schemas: List[dict]) -> List[types.Tool]:
    """Convert tool definitions to Google Tool schema"""

    # if not isinstance(tools[0], types.Tool):
    #     tools = [types.Tool(function_declarations=[tool]) for tool in tools]
    google_tools = []
    for sch in func_schemas:
        if isinstance(sch, types.Tool):
            google_tools.append(sch)
            continue
        if "type" in sch:
            # remove type field (used by openai)
            sch = sch.copy()
            sch.pop("type")

        # function_declarations = [types.FunctionDeclaration(**tool)]
        google_tool = types.Tool(function_declarations=[sch])
        google_tools.append(google_tool)
    return google_tools


def _prepare_google_config(params: CommonQueryParameters, **kwargs) -> tuple:
    """Prepare config and contents for Google API calls"""
    instructions = params["instructions"]
    documents = params["documents"]
    llm = params["llm"]
    text_format = params.get("text_format")
    tools = params.get("tools")

    contents = _fix_docs_for_google(documents)

    config = kwargs.copy()
    if instructions:
        config["system_instruction"] = instructions

    if text_format is not None:
        config["response_mime_type"] = "application/json"
        config["response_schema"] = text_format

    if tools:
        config["tools"] = _prepare_tool_schema(tools)

    model_name = llm.model_name if llm else "gemini-2.5-flash"

    return model_name, contents, config


class GoogleProvider(BaseProvider):
    provider_type: str = "google"

    def get_default_llm_identity(self) -> LLMIdentity:
        return LLMIdentity("gemini-2.5-flash", provider=self.provider_type)

    def parse_response(self, raw_response: Union[BaseModel, dict]) -> ParsedResponse:
        """Parse Gemini API response into common format"""
        if isinstance(raw_response, BaseModel):
            # Pydantic model (e.g., google.genai.types.GenerateContentResponse)

            response: types.GenerateContentResponse = raw_response
            text = response.text
            response_id = response.response_id
            obj = response.model_dump(mode="json")
            obj.pop("response_id", None)

            tools = []
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    func_call: types.FunctionCall = part.function_call
                    tools.append(
                        (
                            func_call.name,
                            # a bit annoying because we're doing str -> dict -> str -> dict
                            # json.dumps(func_call.args),
                            func_call.args,
                            func_call.id,
                        )
                    )
            if tools and text is None:
                text = ""  # I guess this can happen
            return ParsedResponse(
                text=text, response_id=response_id, metadata=obj, tool_calls=tools
            )
        elif isinstance(raw_response, dict):
            # Dict response
            resp_id = raw_response.pop("response_id", None) or raw_response.pop(
                "responseId", None
            )

            # Extract text from Gemini response
            text_content = raw_response.get("text", str(raw_response))

            tools = []
            for part in (
                raw_response.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [])
            ):
                if "function_call" in part:
                    func_call = part["function_call"]
                    tools.append(
                        (
                            func_call["name"],
                            func_call["args"],
                            func_call["id"],
                        )
                    )

            if tools and text_content is None:
                text_content = ""  # I guess this can happen
            return ParsedResponse(
                text=text_content,
                response_id=resp_id,
                metadata=raw_response,
                tool_calls=tools,
            )
        else:
            raise ValueError(f"Unsupported response type: {type(raw_response)}")


class SyncGoogleProvider(SyncProvider, GoogleProvider):
    def __init__(self, client: "genai.Client"):
        self.client = client

    def prepare_sync_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare a synchronous callable for Gemini API"""
        model_name, contents, config = _prepare_google_config(params, **kwargs)

        return self.client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )


class AsyncGoogleProvider(AsyncProvider, GoogleProvider):
    def __init__(self, client: "genai.Client"):
        self.client = client

    def prepare_async_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare an async coroutine for Gemini API"""
        model_name, contents, config = _prepare_google_config(params, **kwargs)

        coro = self.client.aio.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        return coro
