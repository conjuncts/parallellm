
TODO make seq_id not necessarily auto-increment

explicitly not-agentic philosophy (more of an input/output machine) although agents / responsibility isolation can be implemented with LLMIdentity


- [x] different execution counters
- [x] condition_hash (salt-by)
- [ ] batch api
- [ ] allow LLM to change on a per-`ask_llm` level
    - concoct a "multi-provider" that routes based on `provider_type`
- [ ] dedicated SQLite storage for requests that error
- [ ] retrieve() should also be able to return if a value is pending (in addition to present/absent)
- [ ] "cohort locking": for consistency, if seq_id is "strict" (if we really care that seq_id is consistent across runs), then we need to "lock" based on cohort (wait until all batches in a cohort complete. This can be implemented simply by refusing to proceed - ie. ).
    - this is like a rendezvous in threading
- [x] Automatically persist upon pllm exit
- [ ] accept dict as a LLMDocument


## TODO
- centrally track documents (and incorporate with MessageState) just as responses are also tracked
- tree-based MessageState, which in turn stores all historical MessageState's

- Error handling: 
    - sync: ask_llm raises an error OR ask_llm produces an error object (ErrorResponse), which is raised when resolve() is called
    - async: ask_llm is fine, but resolve() raises an error
    - mode 1: exceptions are fatal
    - mode 2: log exceptions and continue
    - three error handling modes: None, skip, retry (exponential backoff)
    - Better errors when a bad request (ie. openai.BadRequest) is made



Input storage:
- [x] doc_hash <=> list of message hashes (doc_table)
- [x] message_hash <=> message_value (message_table)
- [x] tool calls for batch mode
- [x] image as valid document type
- [ ] fix tag for batches
- [ ] roll up messages when several consecutive come from the same role

- [ ] the doc_hash/msg_hash naming convention is kinda backward due to history

- [ ] resolve_all
- [ ] export_all
- [ ] import batch.zip
