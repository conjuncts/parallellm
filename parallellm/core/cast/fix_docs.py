from typing import List, Optional, Union

from parallellm.types import FunctionCallRequest, LLMDocument, LLMResponse


def _to_assistant_message(resp: LLMResponse) -> LLMDocument:
    """Converts LLMResponse back into a LLMDocument."""
    val = resp.resolve()
    if resp._pr and resp._pr.function_calls:
        return FunctionCallRequest(
            text_content=val,
            calls=resp.resolve_function_calls(to_dict=False),
            call_id=resp.call_id,
        )
    return ("assistant", val)


def cast_documents(
    documents: Union[
        Union[LLMDocument, LLMResponse], List[Union[LLMDocument, LLMResponse]]
    ],
    additional_documents: Optional[List[Union[LLMDocument, LLMResponse]]] = None,
) -> List[LLMDocument]:
    """
    Cast a list of documents to standardize to a common type.
    """
    if not isinstance(documents, list):
        documents = [documents]
    if additional_documents is None:
        additional_documents = []
    result = documents + additional_documents
    for i, item in enumerate(result):
        if isinstance(item, LLMResponse):
            result[i] = _to_assistant_message(item)
    return result
