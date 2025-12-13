from typing import List, Optional, Union

from parallellm.core.response import LLMResponse
from parallellm.types import LLMDocument


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
    return result
