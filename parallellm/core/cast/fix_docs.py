from typing import List, Optional, Union

from parallellm.core.response import LLMDocument


def cast_documents(
    documents: Union[LLMDocument, List[LLMDocument]],
    additional_documents: Optional[List[LLMDocument]] = None,
) -> List[LLMDocument]:
    """
    Cast a list of documents to standardize to a common type.
    """
    if not isinstance(documents, list):
        documents = [documents]
    if additional_documents is None:
        additional_documents = []
    return documents + additional_documents
