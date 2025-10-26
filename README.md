Parallellm

(Parallel Language Models)

TODO make seq_id not necessarily auto-increment

Better errors when a bad request (ie. openai.BadRequest) is made

explicitly not-agentic philosophy (more of an input/output machine) although agents / responsibility isolation can be implemented with LLMIdentity


- [x] difference between checkpoint-control and non-checkpoint-control
- [x] different execution counters
- [x] condition_hash (salt-by)
- [ ] batch api
- [ ] allow LLM to change on a per-`ask_llm` level
    - concoct a "multi-provider" that routes based on `provider_type`
- [ ] dedicated SQLite storage for requests that error
- [ ] retrieve() should also be able to return if a value is pending (in addition to present/absent)
- [ ] right now, (for batch) bookkeep_call() is the one that emits an error if a value is pending but not available, but this seems unelegant and should probably be moved out
- [ ] "cohort locking": for consistency, if seq_id is "strict" (if we really care that seq_id is consistent across runs), then we need to "lock" based on cohort (wait until all batches in a cohort complete. This can be implemented simply by refusing to proceed - ie. ).
    - this is like a rendezvous in threading
- [x] Automatically persist upon pllm exit
- [ ] accept dict as a LLMDocument
- [ ]

| Default | Sync | Async | Batch |
| --- | --- | --- | --- |
| OpenAI | ✅ | ✅ | ✅ |
| Anthropic | ✅ | ✅ | TODO |
| Google | ✅ | ✅ | TODO |


| Structured Output | Sync | Async | Batch |
| --- | --- | --- | --- |
| OpenAI | ✅ | ✅ | ✅? |
| Anthropic | ❌ | ❌ | ❌ |
| Google | ✅ | ✅ | TODO |

| Tool Calls | Sync | Async | Batch |
| --- | --- | --- | --- |
| OpenAI | ✅ | ✅ | TODO |
| Anthropic | ✅ | ✅ | TODO |
| Google | ✅ | ✅ | TODO |
