from typing import Literal, Union
from parallellm.core.response import LLMResponse
from parallellm.types import (
    DocumentType,
    LLMDocument,
    FunctionCall,
    FunctionCallOutput,
    FunctionCallRequest,
)


def cast_document_to_str(
    doc: Union[LLMDocument, LLMResponse],
) -> tuple[str, DocumentType, str]:
    """
    Convert Document to string, for purpose of serialization.

    :param doc: The LLMDocument to cast
    :return: Tuple of (doc_value: str, doc_type: str, doc_extra: str)
    """
    if isinstance(doc, str):
        return doc, "text", None
    elif isinstance(doc, FunctionCallRequest):
        # This is an assistant message that should be stored in the datastore anyway.
        # TODO: determine whether this should be serialized.
        # TODO: you could probably store the unique identifier instead
        # ({agent_name}:{seq_id}:{sess_id})
        pass
    elif isinstance(doc, FunctionCallOutput):
        return doc.content, "function_call_output", doc.call_id
    elif isinstance(doc, tuple):
        return doc[1], "text", doc[0]
    else:
        raise NotImplementedError(f"Unknown document type: {type(doc)}")
