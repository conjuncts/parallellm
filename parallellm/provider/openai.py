import json
from typing import TYPE_CHECKING, List, Optional, Union
from pydantic import BaseModel
from parallellm.core.identity import LLMIdentity
from parallellm.provider.base import (
    AsyncProvider,
    BaseProvider,
    BatchProvider,
    SyncProvider,
)
from parallellm.provider.openai_tools import to_strict_json_schema
from parallellm.provider.schemas import guess_schema
from parallellm.types import (
    BatchIdentifier,
    BatchResult,
    BatchStatus,
    CallIdentifier,
    CommonQueryParameters,
    LLMDocument,
    ParsedResponse,
)

if TYPE_CHECKING:
    from openai import OpenAI, AsyncOpenAI
    from openai.types.responses.response_input_param import Message
    from openai.types.responses.response import Response


class OpenAIProvider(BaseProvider):
    provider_type: str = "openai"

    def _fix_docs_for_openai(
        self,
        documents: Union[LLMDocument, List[LLMDocument]],
    ) -> "List[Message]":
        """Ensure documents are in the correct format for OpenAI API"""
        if not isinstance(documents, list):
            documents = [documents]

        documents = documents.copy()
        for i, doc in enumerate(documents):
            if isinstance(doc, str):
                msg: "Message" = {
                    "role": "user",
                    "content": doc,
                }
                documents[i] = msg
            elif isinstance(doc, tuple) and len(doc) == 2:
                # Handle Tuple[Literal["user", "assistant", "system", "developer"], str]
                role, content = doc
                msg: "Message" = {
                    "role": role,
                    "content": content,
                }
                documents[i] = msg
        return documents

    def get_default_llm_identity(self) -> LLMIdentity:
        return LLMIdentity("gpt-4.1-nano", provider=self.provider_type)

    def parse_response(self, raw_response: Union[BaseModel, dict]) -> ParsedResponse:
        """Parse OpenAI API response into common format"""
        if isinstance(raw_response, BaseModel):
            # Pydantic model (e.g., from openai.types.responses.response.Response)
            response: Response = raw_response
            text = response.output_text
            obj = response.model_dump(mode="json")
            resp_id = response.id
            obj.pop("id", None)

            tool_calls = []
            for item in response.output:
                if item.type == "function_call":
                    tool_calls.append((item.name, item.arguments, item.call_id))
                elif item.type == "custom_tool_call":
                    tool_calls.append((item.name, item.input, item.call_id))

            parsed_metadata = obj
        elif isinstance(raw_response, dict):
            # Dict response (e.g., from batch API)

            if "output_text" in raw_response:
                # Used in testing
                text = raw_response["output_text"]
                tool_calls = []
            else:
                # Extract text from OpenAI responses API format
                tool_calls = []
                texts: List[str] = []
                for output in raw_response.get("output", []):
                    if output["type"] == "message":
                        for content in output["content"]:
                            if content["type"] == "output_text":
                                texts.append(content["text"])
                            elif content["type"] == "function_call":
                                tool_calls.append(
                                    (
                                        content["name"],
                                        content["arguments"],
                                        content.get("call_id"),
                                    )
                                )
                            elif content["type"] == "custom_tool_call":
                                tool_calls.append(
                                    (
                                        content["name"],
                                        content["input"],
                                        content.get("call_id"),
                                    )
                                )
                text = "".join(texts)

            resp_id = raw_response.pop("id", None)
            parsed_metadata = raw_response
        else:
            raise ValueError(f"Unsupported response type: {type(raw_response)}")
        return ParsedResponse(
            text=text,
            response_id=resp_id,
            metadata=parsed_metadata,
            tool_calls=tool_calls,
        )


class SyncOpenAIProvider(SyncProvider, OpenAIProvider):
    def __init__(self, client: "OpenAI"):
        self.client = client

    def prepare_sync_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare a synchronous callable for OpenAI API"""
        instructions = params["instructions"]
        documents = self._fix_docs_for_openai(params["documents"])
        llm = params["llm"]
        text_format = params.get("text_format")

        if text_format is not None:
            return self.client.responses.parse(
                model=llm.model_name,
                instructions=instructions,
                input=documents,
                text_format=text_format,
                **kwargs,
            )

            # if "text" not in kwargs:
            #     kwargs["text"] = {}

            # schema = to_strict_json_schema(text_format)
            # kwargs["text"]["format"] = {
            #     "type": "json_schema",
            #     "strict": True,
            #     "name": schema.get("title", "UnknownSchema"),
            #     "schema": schema,
            # }
        return self.client.responses.create(
            model=llm.model_name,
            instructions=instructions,
            input=documents,
            **kwargs,
        )


class AsyncOpenAIProvider(AsyncProvider, OpenAIProvider):
    def __init__(self, client: "AsyncOpenAI"):
        self.client = client

    def prepare_async_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare an async coroutine for OpenAI API"""
        instructions = params["instructions"]
        documents = self._fix_docs_for_openai(params["documents"])
        llm = params["llm"]
        text_format = params.get("text_format")

        if text_format is not None:
            coro = self.client.responses.parse(
                model=llm.model_name,
                instructions=instructions,
                input=documents,
                text_format=text_format,
                **kwargs,
            )
        else:
            coro = self.client.responses.create(
                model=llm.model_name,
                instructions=instructions,
                input=documents,
                **kwargs,
            )

        return coro


class BatchOpenAIProvider(BatchProvider, OpenAIProvider):
    def __init__(self, client: "OpenAI"):
        self.client = client

    def _turn_to_openai_batch(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        instructions = params["instructions"]
        fixed_documents = self._fix_docs_for_openai(params["documents"])
        llm = params["llm"]
        text_format = params.get("text_format")

        if text_format is not None:
            if "text" not in kwargs:
                kwargs["text"] = {}

            assert not kwargs["text"].get("format"), (
                "Cannot supply both text_format and text.format"
            )
            schema = to_strict_json_schema(text_format)
            kwargs["text"]["format"] = {
                "type": "json_schema",
                "strict": True,
                "name": schema.get("title", "UnknownSchema"),
                "schema": schema,
            }

        body = {
            "model": llm.model_name,
            "instructions": instructions,
            "input": fixed_documents,
            **kwargs,
        }
        return {"method": "POST", "url": "/v1/responses", "body": body}

    def prepare_batch_call(
        self,
        params: CommonQueryParameters,
        **kwargs,
    ):
        """Prepare batch call data for OpenAI"""
        return self._turn_to_openai_batch(
            params,
            **kwargs,
        )

    def _decode_openai_batch_result(self, result: dict) -> ParsedResponse:
        """Decode a single result from OpenAI batch response"""
        custom_id = result["custom_id"]

        body = result["response"]["body"]

        # destroyed metadata:
        # result["error"] (should be null)
        # result["response"]["status_code"] (should be 200)
        # redundant IDs (supplanted by custom_id):
        #   result["id"]
        #   result["response"]["request_id"]
        #   result["response"]["body"]["id"]

        parsed = guess_schema(body, provider_type="openai")
        # Note: custom_id overrides the response_id from the body
        return ParsedResponse(
            text=parsed.text, response_id=custom_id, metadata=parsed.metadata
        )

    def _decode_openai_batch_error(self, result: dict) -> ParsedResponse:
        """Decode a single error from OpenAI batch response

        The text field contains the error code.
        """
        custom_id = result.pop("custom_id", None)
        # destroyed metadata:
        # redundant IDs (see above, supplanted by custom_id)
        # result["response"]["body"]

        err_obj = result.get("error", {})
        # actually, error_obj is usually None, and the real error is in response.body.error

        resp_obj = result.get("response", {})
        error_code = resp_obj.get("status_code")
        if error_code is not None:
            error_code = str(error_code)

        if err_obj is not None:
            # unusually, the error is indeed here
            return ParsedResponse(
                text=error_code or "", response_id=custom_id, metadata=err_obj
            )

        # need to retrieve error from response body
        body = resp_obj.get("body", {})
        body_error = body.get("error", {})
        return ParsedResponse(
            text=error_code or "", response_id=custom_id, metadata=body_error
        )

    def submit_batch_to_provider(
        self, call_ids: list[CallIdentifier], stuff: list[dict]
    ) -> BatchIdentifier:
        """
        Submit a batch of calls to the provider.

        This is called from the backend.
        """
        fpath = self.backend.persist_batch_to_jsonl(stuff)
        with open(fpath, "rb") as f:
            batch_input_file = self.client.files.create(file=f, purpose="batch")
            batch_input_file_id = batch_input_file.id

        batch_obj = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/responses",
            completion_window="24h",
            # metadata={"description":""}
        )

        # Extract custom_ids from stuff (they were added by persist_batch_to_jsonl)
        custom_ids = [s["custom_id"] for s in stuff]

        return BatchIdentifier(
            call_ids=call_ids,
            custom_ids=custom_ids,
            batch_uuid=batch_obj.id,
        )

    def download_batch(
        self,
        batch_uuid: str,
    ) -> List[BatchResult]:
        """Download the results of a batch from the provider.

        :param batch_uuid: The UUID of the batch to download.
        :return: List of BatchResult objects containing the results and errors (if any).
            If nothing is ready yet, empty list is returned.
        """
        batch = self.client.batches.retrieve(batch_uuid)
        err_file_id = batch.error_file_id
        out_file_id = batch.output_file_id

        if out_file_id is None and err_file_id is None:
            return []

        results = []

        # Successful completion
        out_content = self.client.files.content(out_file_id).text
        try:
            parsed_results = [
                self._decode_openai_batch_result(json.loads(line))
                for line in out_content.strip().split("\n")
                if line
            ]
            suc_res = BatchResult(
                status="ready",
                raw_output=out_content,
                parsed_responses=parsed_results,
            )
        except json.JSONDecodeError:
            suc_res = BatchResult(
                status="error",
                raw_output=out_content,
                parsed_responses=None,
            )
        results.append(suc_res)

        # Errors
        if err_file_id is not None:
            err_content = self.client.files.content(err_file_id).text

            try:
                parsed_errors = [
                    self._decode_openai_batch_error(json.loads(line))
                    for line in err_content.strip().split("\n")
                    if line
                ]
                err_res = BatchResult(
                    status="error",
                    raw_output=err_content,
                    parsed_responses=parsed_errors,
                )
            except json.JSONDecodeError:
                err_res = BatchResult(
                    status="error",
                    raw_output=err_content,
                    parsed_responses=None,
                )
            results.append(err_res)
        return results
