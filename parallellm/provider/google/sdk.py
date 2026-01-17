import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union
from venv import logger
from pydantic import BaseModel
from parallellm.core.identity import LLMIdentity
from parallellm.file_io.file_manager import FileManager
from parallellm.provider.base import (
    AsyncProvider,
    BaseProvider,
    BatchProvider,
    SyncProvider,
)
from parallellm.types import (
    BatchIdentifier,
    BatchResult,
    CallIdentifier,
    CommonQueryParameters,
    LLMDocument,
    ParsedResponse,
    ToolCall,
    ToolCallOutput,
    ToolCallRequest,
)

from google import genai
from google.genai import types


def _fix_docs_for_google(
    documents: Union[LLMDocument, List[LLMDocument]],
) -> List[Union[dict, types.Content]]:
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
        elif isinstance(doc, ToolCallOutput):
            # https://ai.google.dev/gemini-api/docs/function-calling?example=meeting
            function_response_part = types.Part(
                function_response=types.FunctionResponse(
                    name=doc.name,
                    response={"output": doc.content},
                )
            )
            formatted_docs.append(
                types.Content(role="user", parts=[function_response_part])
            )
        elif isinstance(doc, ToolCallRequest):
            parts = []
            if doc.text_content:
                parts.append(types.Part(text=types.Text(content=doc.text_content)))
            parts += [
                types.Part(
                    function_call=types.FunctionCall(
                        name=call.name,
                        args=call.args,
                        id=call.call_id,
                    )
                )
                for call in doc.calls
            ]
            formatted_docs.append(
                types.Content(
                    role="model",
                    parts=parts,
                )
            )
        elif isinstance(doc, tuple) and len(doc) == 2:
            # from google.genai.types import Content
            # For google, only valid roles are ["user", "model"]

            role, content = doc
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
            else:
                raise ValueError(f"Invalid document dict format for Google: {doc}")
        else:
            raise ValueError(f"Unsupported document type: {type(doc)}")

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


def _prepare_google_config(params: CommonQueryParameters, **kwargs):
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


def _extract_text_from_gemini_model(resp: BaseModel):
    """Extract text content from Gemini API response.
    See google.genai.types.GenerateContentResponse._get_text
    Also suppresses a warning
    """
    if (
        not resp.candidates
        or not resp.candidates[0].content
        or not resp.candidates[0].content.parts
    ):
        return None
    text = ""
    any_text_part_text = False
    for part in resp.candidates[0].content.parts:
        if isinstance(part.text, str):
            if isinstance(part.thought, bool) and part.thought:
                continue
            any_text_part_text = True
            text += part.text
    # part.text == '' is different from part.text is None
    return text if any_text_part_text else None


def _extract_text_from_gemini_dict(resp: dict):
    """Extract text content from Gemini API response.
    See google.genai.types.GenerateContentResponse._get_text
    Also suppresses a warning
    """
    if (
        not resp.get("candidates")
        or not resp["candidates"][0].get("content")
        or not resp["candidates"][0]["content"].get("parts")
    ):
        return None
    text = ""
    any_text_part_text = False
    for part in resp["candidates"][0]["content"]["parts"]:
        if isinstance(part.get("text"), str):
            if isinstance(part.get("thought"), bool) and part["thought"]:
                continue
            any_text_part_text = True
            text += part["text"]
    # part.text == '' is different from part.text is None
    return text if any_text_part_text else None


class GoogleProvider(BaseProvider):
    provider_type: str = "google"

    def get_default_llm_identity(self) -> LLMIdentity:
        return LLMIdentity("gemini-2.5-flash", provider=self.provider_type)

    def parse_response(self, raw_response: Union[BaseModel, dict]) -> ParsedResponse:
        """Parse Gemini API response into common format"""
        if isinstance(raw_response, BaseModel):
            # Pydantic model (e.g., google.genai.types.GenerateContentResponse)

            response: types.GenerateContentResponse = raw_response
            # Suppress warning
            # text = response.text
            text = _extract_text_from_gemini_model(response)
            response_id = response.response_id
            obj = response.model_dump(mode="json")
            obj.pop("response_id", None)

            tools = []

            if not response.candidates:
                # Could be an error like rate limit
                pass

            for part in response.candidates[0].content.parts or []:
                if part.function_call:
                    func_call: types.FunctionCall = part.function_call
                    tools.append(
                        ToolCall(
                            name=func_call.name,
                            arguments=func_call.args,
                            call_id=func_call.id,
                        )
                    )
            if tools and text is None:
                text = ""  # I guess this can happen
            return ParsedResponse(
                text=text, response_id=response_id, metadata=obj, tool_calls=tools
            )
        elif isinstance(raw_response, dict):
            resp_id = raw_response.pop("response_id", None) or raw_response.pop(
                "responseId", None
            )

            # Extract text from Gemini response
            # from google.genai.types.GenerateContentResponse import _get_text
            text_content = _extract_text_from_gemini_dict(raw_response)

            tools = []
            for part in (
                raw_response.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [])
            ):
                if "function_call" in part:
                    func_call = part["function_call"]
                    tools.append(
                        ToolCall(
                            name=func_call["name"],
                            arguments=func_call["args"],
                            call_id=func_call["id"],
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

        # from google.genai.errors import ClientError

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


class BatchGoogleProvider(BatchProvider, GoogleProvider):
    def __init__(self, client: "genai.Client"):
        self.client = client

    def _turn_to_gemini_batch(
        self,
        params: CommonQueryParameters,
        custom_id: str,
        **kwargs,
    ):
        """Convert CommonQueryParameters to Gemini batch request format"""
        instructions = params["instructions"]
        fixed_documents = _fix_docs_for_google(params["documents"])
        llm = params["llm"]
        text_format = params.get("text_format")
        tools = params.get("tools")

        gen_config = kwargs.copy()
        config = {}
        if instructions:
            config["system_instruction"] = instructions

        if text_format is not None:
            gen_config["responseMimeType"] = "application/json"
            if not isinstance(text_format, dict):
                # pydantic?
                if getattr(text_format, "model_json_schema", None):
                    text_format = text_format.model_json_schema()
            gen_config["responseJsonSchema"] = text_format

        if tools:
            gen_config["tools"] = _prepare_tool_schema(tools)

        # Gemini batch request format
        request = {
            "key": custom_id,
            "request": {
                "contents": fixed_documents
                if isinstance(fixed_documents, list)
                else [{"parts": [{"text": fixed_documents}], "role": "user"}],
                **config,
                "generationConfig": gen_config,
            },
        }

        return request

    def prepare_batch_call(
        self,
        params: CommonQueryParameters,
        custom_id: str,
        **kwargs,
    ):
        """Prepare batch call data for Gemini"""
        return self._turn_to_gemini_batch(
            params,
            custom_id=custom_id,
            **kwargs,
        )

    def _decode_gemini_batch_result(
        self, result: dict, custom_id: str
    ) -> ParsedResponse:
        """Decode a single result from Gemini batch response"""

        # For successful responses, the result should be a GenerateContentResponse
        if "response" in result:
            response_data = result["response"]
            # Use existing parse_response method to handle the response
            parsed = self.parse_response(response_data)
            # Override response_id with custom_id
            return ParsedResponse(
                text=parsed.text,
                response_id=custom_id,
                metadata=parsed.metadata,
                tool_calls=parsed.tool_calls,
            )
        else:
            # Fallback parsing
            return ParsedResponse(
                text=result.get("text", ""),
                response_id=custom_id,
                metadata=result,
                tool_calls=[],
            )

    def _decode_gemini_batch_error(
        self, result: dict, custom_id: str
    ) -> ParsedResponse:
        """Decode a single error from Gemini batch response"""

        error_info = result.get("error", {})
        error_message = error_info.get("message", "Unknown error")

        return ParsedResponse(
            text=error_message,
            response_id=custom_id,
            metadata=error_info,
            tool_calls=[],
        )

    def get_batch_custom_ids(self, stuff):
        custom_ids = []
        for item in stuff:
            if "key" not in item:
                raise ValueError("Each batch item must have a 'key' field.")
            custom_ids.append(item["key"])
        return custom_ids

    def submit_batch_to_provider(self, fpath: Path, llm: str) -> str:
        """
        Submit a batch of calls to the provider.

        This is called from the backend.
        """
        from google.genai import types

        num_lines = 0
        with open(fpath, "r", encoding="utf-8") as f:
            for _ in f:
                num_lines += 1
        # Upload the file to Gemini File API
        uploaded_file = self.client.files.upload(
            file=fpath,
            config=types.UploadFileConfig(
                display_name=f"batch-requests-{num_lines}",
                mime_type="application/json",
            ),
        )

        # Create batch job
        batch_job = self.client.batches.create(
            model=llm,
            src=uploaded_file.name,
            config={
                "display_name": f"batch-job-{num_lines}-requests",
            },
        )
        return batch_job.name.removeprefix("batches/")

        # TODO: consider: a try/finally block option to clean up the temp files?

    def download_batch(
        self,
        batch_uuid: str,
    ) -> List[BatchResult]:
        """Download the results of a batch from the provider.

        :param batch_uuid: The UUID of the batch to download.
        :return: List of BatchResult objects containing the results and errors (if any).
            If nothing is ready yet, empty list is returned.
        """
        import json

        # Get batch job status
        batch_job = self.client.batches.get(name="batches/" + batch_uuid)

        # Check if job is completed
        completed_states = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        }

        if batch_job.state.name not in completed_states:
            return []  # Still pending

        results = []

        if batch_job.state.name == "JOB_STATE_SUCCEEDED":
            if batch_job.dest and batch_job.dest.file_name:
                # Results are in a file
                try:
                    file_content = self.client.files.download(
                        file=batch_job.dest.file_name
                    )
                    content_str = file_content.decode("utf-8")

                    parsed_responses = []
                    for line in content_str.strip().split("\n"):
                        if line:
                            try:
                                line_data = json.loads(line)
                                custom_id = line_data.get("key", "unknown")

                                if "response" in line_data:
                                    parsed_response = self._decode_gemini_batch_result(
                                        line_data, custom_id
                                    )
                                else:
                                    parsed_response = self._decode_gemini_batch_error(
                                        line_data, custom_id
                                    )

                                parsed_responses.append(parsed_response)
                            except json.JSONDecodeError:
                                # Handle malformed JSON lines
                                continue

                    results.append(
                        BatchResult(
                            status="ready",
                            raw_output=content_str,
                            parsed_responses=parsed_responses,
                        )
                    )
                except Exception as e:
                    results.append(
                        BatchResult(
                            status="error",
                            raw_output=str(e),
                            parsed_responses=None,
                        )
                    )
            else:
                raise ValueError("Batch job succeeded but no destination file found.")

        else:
            # Job failed, cancelled, or expired
            error_msg = getattr(
                batch_job, "error", f"Job state: {batch_job.state.name}"
            )
            results.append(
                BatchResult(
                    status="error",
                    raw_output=str(error_msg),
                    parsed_responses=None,
                )
            )

        return results
