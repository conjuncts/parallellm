import json
from typing import List, Tuple

from parallellm.types import ToolCall


def dump_tool_calls(tool_calls: List[ToolCall]) -> str:
    """
    Serialize a list of ToolCall objects into a JSON-ified list of tuples.
    """
    return json.dumps([[call.name, call.args, call.call_id] for call in tool_calls])


def load_tool_calls(data: str) -> List[ToolCall]:
    """
    Deserialize a list of tuples into a list of ToolCall objects.
    """
    tuples = json.loads(data)
    return [
        ToolCall(name=name, arguments=arguments, call_id=call_id)
        for name, arguments, call_id in tuples
    ]
