# Parallellm

(Parallel Language Models)



## 

TODO make seq_id not necessarily auto-increment

Better errors when a bad request (ie. openai.BadRequest) is made

explicitly not-agentic philosophy (more of an input/output machine) although agents / responsibility isolation can be implemented with LLMIdentity


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

- [ ] retry mechanism
- [ ] switch hashes from base16 to base64

| Default | Sync | Async | Batch |
| --- | --- | --- | --- |
| OpenAI | ✅ | ✅ | ✅ |
| Anthropic | ✅ | ✅ | TODO |
| Google | ✅ | ✅ | ✅ |


| Structured Output | Sync | Async | Batch |
| --- | --- | --- | --- |
| OpenAI | ✅ | ✅ | ✅? |
| Anthropic | ❌ | ❌ | ❌ |
| Google | ✅ | ✅ | ✅ |

| Tool Calls | Sync | Async | Batch |
| --- | --- | --- | --- |
| OpenAI | ✅ | ✅ | TODO |
| Anthropic | ✅ | ✅ | TODO |
| Google | ✅ | ✅ | TODO |


## TODO
- centrally track documents (and incorporate with MessageState) just as responses are also tracked
- tree-based MessageState, which in turn stores all historical MessageState's

- Error handling: 
    - sync: ask_llm raises an error OR ask_llm produces an error object (ErrorResponse), which is raised when resolve() is called
    - async: ask_llm is fine, but resolve() raises an error
    - mode 1: exceptions are fatal
    - mode 2: log exceptions and continue

## Philosophy

1. A library designed with the Batch API in mind.
2. We aim to support pipelines where LLMs are "input/output machines", rather than an interactive conversational agent.
3. LLM pipeline control flow should be represented with Python, rather than data structures (ie. LangChain, LangGraph).
4. Circumvent Vendor Lock-in and "Architecture Lock-in".
5. Effortless parallelization.
6. Improved Developer Experience (Develop and debug as synchronous, quickly scale up to huge pipelines).