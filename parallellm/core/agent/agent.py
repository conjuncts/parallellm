from typing import TYPE_CHECKING, List, Optional, Union
from colorama import Fore, Style, init
from parallellm.core.cast.fix_docs import cast_documents
from parallellm.core.exception import GotoCheckpoint, NotAvailable, WrongCheckpoint
from parallellm.core.hash import compute_hash
from parallellm.core.response import (
    LLMIdentity,
    LLMResponse,
    ReadyLLMResponse,
)
from parallellm.logging.dash_logger import HashStatus
from parallellm.types import CallIdentifier, LLMDocument

if TYPE_CHECKING:
    from parallellm.core.agent.orchestrator import AgentOrchestrator


class AgentContext:
    """Context manager for Agent lifecycle (default context)"""

    def __init__(self, agent_name, batch_manager: "AgentOrchestrator"):
        self.agent_name = agent_name
        self._bm = batch_manager

        self._anonymous_counter = 0  # Anonymous mode counter, always starts from 0
        self._checkpoint_counter = None  # Initialized when entering checkpoint
        self.active_checkpoint: Optional[str] = None
        """
        Active checkpoint: if we are inside a checkpoint context.
        Either a checkpoint or None (if anonymous).
        """

    def __enter__(self):
        # Any setup logic can go here if needed
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Delegate to BatchManager's error handling logic

        self._exit_checkpoint()

        if exc_type in (NotAvailable, WrongCheckpoint, GotoCheckpoint):
            return True
        return False

    def print(self, *args, **kwargs):
        """
        Print to console above the dashboard output.
        This ensures proper display ordering when the dashboard is active.
        """
        # Use the dashboard logger's coordinated print method
        print(*args, **kwargs)

    @property
    def my_metadata(self) -> dict:
        """
        Metadata for this agent.
        Backed by the AgentOrchestrator's FileManager.
        """
        key = self.agent_name if self.agent_name is not None else "default-agent"
        return self._bm._fm.metadata["agents"].setdefault(
            key,
            {
                "latest_checkpoint": None,
                "checkpoint_counter": 0,
            },
        )

    def when_checkpoint(self, checkpoint_name):
        # Always allow enter if no checkpoints yet
        if self.my_metadata["latest_checkpoint"] is None:
            self.my_metadata["latest_checkpoint"] = checkpoint_name

        # But generally, only enter if it is the latest
        elif checkpoint_name != self.my_metadata["latest_checkpoint"]:
            raise WrongCheckpoint()

        # Set it as active and initialize local checkpoint counter
        self.active_checkpoint = checkpoint_name
        # Initialize local counter from persisted metadata (or 0 if first time)
        self._checkpoint_counter = self.my_metadata.get("checkpoint_counter", 0)

        self._bm._logger.info(
            f"Entered checkpoint {Fore.CYAN}{checkpoint_name}{Style.RESET_ALL}"
        )

    def goto_checkpoint(self, checkpoint_name):
        """
        Nothing after this statement will be executed.
        """
        # Log the current seq_id before switching
        current_seq_id = None
        if self.active_checkpoint is not None:
            current_seq_id = self._checkpoint_counter
        else:
            current_seq_id = self._anonymous_counter

        # Revise metadata
        self.my_metadata["latest_checkpoint"] = checkpoint_name
        self.my_metadata["checkpoint_counter"] = self._checkpoint_counter

        # Logging
        self._bm._fm.log_checkpoint_event(
            "switch", self.agent_name, checkpoint_name, current_seq_id
        )
        self._bm._logger.info(
            f"Switched to checkpoint {Fore.CYAN}{checkpoint_name}{Style.RESET_ALL}"
        )
        raise GotoCheckpoint(checkpoint_name)

    def _exit_checkpoint(self):
        """
        Exit checkpoint context
        """
        if self.active_checkpoint is not None:
            # Log the final seq_id before exiting
            final_seq_id = (
                self._checkpoint_counter if self._checkpoint_counter is not None else 0
            )

            # Log checkpoint exit to file
            # self._fm.log_checkpoint_event("exit", self.active_checkpoint, final_seq_id)

            # Do NOT persist local checkpoint counter
            # (local checkpoint counter should only persist upon a goto)
            self.active_checkpoint = None
            self._checkpoint_counter = None

    def ask_llm(
        self,
        documents: Union[LLMDocument, List[LLMDocument]],
        *additional_documents: LLMDocument,
        instructions: Optional[str] = None,
        llm: Union[LLMIdentity, str, None] = None,
        _hoist_images=None,
        **kwargs,
    ) -> LLMResponse:
        """
        Ask the LLM a question

        :param documents: Documents to use, such as the prompt.
            Can be strings or images.
        :param instructions: The system prompt to use.
        :param llm: The identity of the LLM to use.
            Can be helpful multi-agent or multi-model scenarios.
        :param _hoist_images: Gemini recommends that images be hoisted to the front of the message.
            Set to True/False to explicitly enforce/disable.
        :returns: A LLMResponse. The value is **lazy loaded**: for best efficiency,
            it should not be resolved until you actually need it.
        """

        if isinstance(llm, str):
            llm = LLMIdentity(llm)

        # Use dual counter system based on checkpoint mode
        if self.active_checkpoint is not None:
            # In checkpoint mode: use and increment local checkpoint counter
            seq_id = self._checkpoint_counter
            self._checkpoint_counter += 1
        else:
            # In anonymous mode: use and increment anonymous counter
            seq_id = self._anonymous_counter
            self._anonymous_counter += 1

        documents = cast_documents(documents, list(additional_documents))

        # Cache using datastore
        hashed = compute_hash(instructions, documents)

        call_id: CallIdentifier = {
            "agent_name": self.agent_name,
            "checkpoint": self.active_checkpoint,
            "doc_hash": hashed,
            "seq_id": seq_id,
            "session_id": self._bm.get_session_counter(),
            "provider_type": self._bm._provider.provider_type,
        }
        cached = self._bm._backend.retrieve(call_id)
        if cached is not None:
            self.update_hash_status(hashed, HashStatus.CACHED)
            return ReadyLLMResponse(
                call_id=call_id,
                value=cached,
            )

        # Not cached, submit to provider
        self.update_hash_status(hashed, HashStatus.SENT)
        return self._bm._provider.submit_query_to_provider(
            instructions,
            documents,
            call_id=call_id,
            llm=llm,
            _hoist_images=_hoist_images,
            **kwargs,
        )

    def update_hash_status(self, hash_value: str, status: HashStatus):
        # No-op
        pass


class AgentDashboardContext(AgentContext):
    """Context manager for the hash status dashboard"""

    def __init__(self, agent_name, batch_manager: "AgentOrchestrator", log_k: int = 10):
        super().__init__(agent_name, batch_manager)
        self._was_displaying = False
        self._bm._dash_logger.k = log_k

    @property
    def _dash_logger(self):
        return self._bm._dash_logger

    def __enter__(self):
        # Store current display state and enable display
        self._was_displaying = self._dash_logger.display
        self._dash_logger.set_display(True)
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Print final status of dashboard
        self._dash_logger._update_console()

        # Finalize the line and restore original display state
        self._dash_logger.finalize_line()
        self._dash_logger.set_display(self._was_displaying)
        print()
        return super().__exit__(exc_type, exc_val, exc_tb)

    def print(self, *args, **kwargs):
        """
        Print to console above the dashboard output.
        This ensures proper display ordering when the dashboard is active.
        """
        # Use the dashboard logger's coordinated print method
        self._dash_logger.coordinated_print(*args, **kwargs)

    def update_hash_status(self, hash_value: str, status: HashStatus):
        """
        Update the status of a hash in the logger

        Args:
            hash_value: The hash value to update
            status: New status - one of 'C' (cached), '↗' (sent), '↘' (received), '✓' (stored)
        """
        if self._dash_logger is not None:
            self._dash_logger.update_hash(hash_value, status)
