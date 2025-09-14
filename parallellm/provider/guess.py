from typing import Union
from pydantic import BaseModel


def guess_schema(inp: Union[BaseModel, dict]) -> tuple[str, str, dict]:
    """

    Guess the schema of the given object and convert it to a dictionary.
    :param obj: The object to guess the schema for. Can be a Pydantic BaseModel or a dictionary.

    :returns: A tuple containing the response text, response ID, and metadata dictionary.
    """
    if isinstance(inp, BaseModel):
        obj = inp.model_dump(mode="json")

    elif isinstance(inp, dict):
        obj = inp.copy()
    else:
        raise ValueError("Invalid input type")

    resp_text = obj.pop("output_text", None)
    if resp_text is None:
        # Try some other common fields?
        resp_text = getattr(inp, "output_text", None)

    resp_id = (
        obj.pop("id", None)
        or obj.pop("response_id", None)
        or obj.pop("responseId", None)
    )
    return resp_text, resp_id, obj
