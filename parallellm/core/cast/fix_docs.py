from typing import List, Optional, Union

from parallellm.core.msg.state import MessageState
from parallellm.core.response import LLMResponse
from parallellm.types import LLMDocument


_LLMDatagram = Union[LLMDocument, LLMResponse]


def cast_documents(
    documents: Union[_LLMDatagram, List[_LLMDatagram], MessageState],
    additional_documents: Optional[List[_LLMDatagram]] = None,
) -> List[LLMDocument]:
    """
    Cast a list of documents to standardize to a common type.
    """
    if isinstance(documents, MessageState):
        documents = list(documents)
    if not isinstance(documents, list):
        documents = [documents]
    if additional_documents is None:
        additional_documents = []
    result = documents + additional_documents
    # Convert LLMResponse into corresponding messages
    for i, doc in enumerate(result):
        if isinstance(doc, LLMResponse):
            result[i] = doc.to_assistant_message()
    return result
