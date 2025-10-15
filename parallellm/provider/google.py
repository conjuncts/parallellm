from typing import TYPE_CHECKING, List, Optional, Tuple, Union
from pydantic import BaseModel
from parallellm.core.identity import LLMIdentity
from parallellm.provider.base import AsyncProvider, BaseProvider, SyncProvider
from parallellm.types import CallIdentifier, LLMDocument

if TYPE_CHECKING:
    from google import genai


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


class GoogleProvider(BaseProvider):
    provider_type: str = "google"

    def get_default_llm_identity(self) -> LLMIdentity:
        return LLMIdentity("gemini-2.5-flash", provider=self.provider_type)

    def parse_response(
        self, raw_response: Union[BaseModel, dict]
    ) -> Tuple[str, Optional[str], dict]:
        """Parse Gemini API response into common format"""
        if isinstance(raw_response, BaseModel):
            # Pydantic model (e.g., google.genai.types.GenerateContentResponse)
            res = raw_response.text, raw_response.response_id
            obj = raw_response.model_dump(mode="json")
            obj.pop("response_id", None)
            return (*res, obj)
        elif isinstance(raw_response, dict):
            # Dict response
            resp_id = raw_response.pop("response_id", None) or raw_response.pop(
                "responseId", None
            )

            # Extract text from Gemini response
            text_content = raw_response.get("text", str(raw_response))

            return text_content, resp_id, raw_response
        else:
            raise ValueError(f"Unsupported response type: {type(raw_response)}")


class SyncGoogleProvider(SyncProvider, GoogleProvider):
    def __init__(self, client: "genai.Client"):
        self.client = client

    def prepare_sync_call(
        self,
        instructions,
        documents: Union[LLMDocument, List[LLMDocument]] = [],
        *,
        llm: LLMIdentity,
        _hoist_images=None,
        **kwargs,
    ):
        """Prepare a synchronous callable for Gemini API"""
        contents = _fix_docs_for_google(documents)

        config = kwargs.copy()
        if instructions:
            config["system_instruction"] = instructions

        def sync_gemini_call():
            return self.client.models.generate_content(
                model=llm.model_name if llm else "gemini-2.5-flash",
                contents=contents,
                config=config,
            )

        return sync_gemini_call


class AsyncGoogleProvider(AsyncProvider, GoogleProvider):
    def __init__(self, client: "genai.Client"):
        self.client = client

    def prepare_async_call(
        self,
        instructions,
        documents: Union[LLMDocument, List[LLMDocument]] = [],
        *,
        llm: LLMIdentity,
        _hoist_images=None,
        **kwargs,
    ):
        """Prepare an async coroutine for Gemini API"""
        contents = _fix_docs_for_google(documents)

        # Prepare generation config with system instructions
        config = kwargs.copy()
        if instructions:
            config["system_instruction"] = instructions

        coro = self.client.aio.models.generate_content(
            model=llm.model_name if llm else "gemini-2.5-flash",
            contents=contents,
            config=config,
        )

        return coro
