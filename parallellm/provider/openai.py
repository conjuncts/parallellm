import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union
from parallellm.core.backend.async_backend import AsyncBackend
from parallellm.core.backend.batch_backend import BatchBackend
from parallellm.core.backend.sync_backend import SyncBackend
from parallellm.core.exception import NotAvailable
from parallellm.core.identity import LLMIdentity
from parallellm.core.response import (
    LLMResponse,
    PendingLLMResponse,
    ReadyLLMResponse,
)
from parallellm.provider.base import (
    AsyncProvider,
    BaseProvider,
    BatchProvider,
    SyncProvider,
)
from parallellm.provider.schemas import guess_schema
from parallellm.types import (
    BatchIdentifier,
    BatchResult,
    BatchStatus,
    CallIdentifier,
    LLMDocument,
)

if TYPE_CHECKING:
    from openai import OpenAI, AsyncOpenAI
    from openai.types.responses.response_input_param import Message


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
        return documents

    def get_default_llm_identity(self) -> LLMIdentity:
        return LLMIdentity("gpt-4.1-nano", provider=self.provider_type)


class SyncOpenAIProvider(SyncProvider, OpenAIProvider):
    def __init__(self, client: "OpenAI", backend: SyncBackend):
        self.client = client
        self.backend = backend

    def submit_query_to_provider(
        self,
        instructions,
        documents: Union[LLMDocument, List[LLMDocument]] = [],
        *,
        call_id: CallIdentifier,
        llm: LLMIdentity,
        _hoist_images=None,
        **kwargs,
    ):
        """Synchronously submit a query to OpenAI and return a ready response"""

        documents = self._fix_docs_for_openai(documents)

        def sync_openai_call():
            return self.client.responses.create(
                model=llm.model_name,
                instructions=instructions,
                input=documents,
                **kwargs,
            )

        # Execute the call synchronously and store the result
        resp_text, _, _ = self.backend.submit_sync_call(
            call_id, sync_function=sync_openai_call
        )

        # Return a ready response since the operation completed immediately
        return ReadyLLMResponse(call_id=call_id, value=resp_text)


class AsyncOpenAIProvider(AsyncProvider, OpenAIProvider):
    def __init__(self, client: "AsyncOpenAI", backend: AsyncBackend):
        self.client = client
        self.backend = backend

    def submit_query_to_provider(
        self,
        instructions,
        documents: Union[LLMDocument, List[LLMDocument]] = [],
        *,
        call_id: CallIdentifier,
        llm: LLMIdentity,
        _hoist_images=None,
        **kwargs,
    ):
        documents = self._fix_docs_for_openai(documents)

        coro = self.client.responses.create(
            model=llm.model_name,
            instructions=instructions,
            input=documents,
            **kwargs,
        )

        # Submit to the backend for asynchronous execution
        self.backend.submit_coro(call_id=call_id, coro=coro)

        return PendingLLMResponse(
            call_id=call_id,
            backend=self.backend,
        )


class BatchOpenAIProvider(BatchProvider, OpenAIProvider):
    def __init__(self, client: "OpenAI", backend: BatchBackend):
        self.client = client
        self.backend = backend
        self.backend._register_provider(self)

    def _turn_to_openai_batch(
        self,
        instructions,
        fixed_documents: List[LLMDocument],
        *,
        llm: LLMIdentity,
        _hoist_images=None,
        **kwargs,
    ):
        body = {
            "model": llm.model_name,
            "instructions": instructions,
            "input": fixed_documents,
            **kwargs,
        }
        return {"method": "POST", "url": "/v1/responses", "body": body}

    def submit_query_to_provider(
        self,
        instructions,
        documents: Union[LLMDocument, List[LLMDocument]] = [],
        *,
        call_id: CallIdentifier,
        llm: Optional[LLMIdentity] = None,
        _hoist_images=None,
        **kwargs,
    ):
        """Submit a query to OpenAI, which (for batch) will never be immediately available"""

        documents = self._fix_docs_for_openai(documents)

        self.backend.bookkeep_call(
            call_id,
            llm,
            stuff=self._turn_to_openai_batch(
                instructions,
                documents,
                llm=llm,
                _hoist_images=_hoist_images,
                **kwargs,
            ),
            auto_assign_id="custom_id",
        )

        # Batch values are always unavailable
        raise NotAvailable()

    def _decode_openai_batch_result(self, result: dict) -> tuple[str, str, dict]:
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

        resp_text, _, resp_meta = guess_schema(body, provider_type="openai")
        return resp_text, custom_id, resp_meta

    def _decode_openai_batch_error(self, result: dict) -> tuple[str, str, dict]:
        """Decode a single error from OpenAI batch response

        The main content (resp_text) is actually the error code.
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
            return error_code, custom_id, err_obj

        # need to retrieve error from response body
        body = resp_obj.get("body", {})
        body_error = body.get("error", {})
        return error_code, custom_id, body_error

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
            oks = [
                self._decode_openai_batch_result(json.loads(line))
                for line in out_content.strip().split("\n")
                if line
            ]
            contents, custom_ids, metadatas = zip(*oks)
            suc_res = BatchResult(
                "ready",
                out_content,
                list(contents),
                list(custom_ids),
                list(metadatas),
            )
        except json.JSONDecodeError:
            suc_res = BatchResult("error", out_content, None, None, None)
        results.append(suc_res)

        # Errors
        if err_file_id is not None:
            err_content = self.client.files.content(err_file_id).text

            try:
                errors = [
                    self._decode_openai_batch_error(json.loads(line))
                    for line in err_content.strip().split("\n")
                    if line
                ]
                contents, custom_ids, metadatas = zip(*errors)
                err_res = BatchResult(
                    "error",
                    err_content,
                    list(contents),
                    list(custom_ids),
                    list(metadatas),
                )
            except json.JSONDecodeError:
                err_res = BatchResult("error", err_content, None, None, None)
            results.append(err_res)
        return results
