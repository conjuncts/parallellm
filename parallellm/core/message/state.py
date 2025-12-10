from collections import UserList
from typing import SupportsIndex, Union
from parallellm.core.response import LLMResponse
from parallellm.types import LLMDocument


class MessageState(UserList[Union[LLMDocument, LLMResponse]]):
    """

    """
    def __init__(self, *, agent_name: str = None, anon_ctr = 0, chkp_ctr = 0):
        super().__init__()
        self.agent_name = agent_name
        self.anon_ctr = anon_ctr
        self.chkp_ctr = chkp_ctr

    def copy(self) -> "MessageState":
        """Create a copy of this MessageState."""
        new_state = MessageState(
            agent_name=self.agent_name,
            anon_ctr=self.anon_ctr,
            chkp_ctr=self.chkp_ctr,
        )
        new_state.data = self.data.copy()
        return new_state

    def _update_seq_counters(self, other: Union[LLMDocument, LLMResponse]):
        """Update sequence counters based on the other message."""
        if isinstance(other, LLMResponse):
            # recover seq_id
            if other.call_id:
                seq_id = other.call_id.get('seq_id', 0)
                if other.call_id.get('checkpoint'):
                    self.chkp_ctr = max(self.chkp_ctr, seq_id)
                else:
                    self.anon_ctr = max(self.anon_ctr, seq_id)

    def __setitem__(self, i, item):
        self._update_seq_counters(item)
        super().__setitem__(i, item)

    def __add__(self, other):
        other_list = []
        if isinstance(other, UserList):
            other_list = other.data
        elif isinstance(other, type(self.data)):
            other_list = other
        else:
            other_list = list(other)

        ret = self.__class__(self.data + other_list)
        for item in other_list:
            ret._update_seq_counters(item)
        return ret

    def __radd__(self, other):
        other_list = []
        if isinstance(other, UserList):
            other_list = other.data
        elif isinstance(other, type(self.data)):
            other_list = other
        else:
            other_list = list(other)
        ret = self.__class__(other_list + self.data)
        for item in other_list:
            ret._update_seq_counters(item)
        return ret

    def __iadd__(self, other):
        other_list = []
        if isinstance(other, UserList):
            other_list = other.data
        elif isinstance(other, type(self.data)):
            other_list = other
        else:
            other_list = list(other)
        self.data += other_list
        for item in other_list:
            self._update_seq_counters(item)
        return self

    def append(self, item: Union[LLMDocument, LLMResponse], /):
        """Append another MessageState to this one and return a new MessageState."""
        self._update_seq_counters(item)
        self.data.append(item)

    def insert(self, i, item):
        self._update_seq_counters(item)
        self.data.insert(i, item)

    def extend(self, others: list[Union[LLMDocument, LLMResponse]], /):
        """Extend this MessageState with a list of other MessageStates."""
        for item in others:
            self._update_seq_counters(item)
        self.data.extend(others)
