Parallellm

(Parallel Language Models)

TODO make seq_id not necessarily auto-increment

Better errors when a bad request (ie. openai.BadRequest) is made

explicitly not-agentic philosophy (more of an input/output machine) although agents / responsibility isolation can be implemented with LLMIdentity


- [x] fix if checkpoint name is invalid name for folder
- [x] store session id
- [x] difference between checkpoint-control and non-checkpoint-control
- [x] different execution counters
- [x] condition_hash (salt-by)
- [ ] batch api
- [ ] allow LLM to change on a per-`ask_llm` level
    - concoct a "multi-provider" that routes based on `provider_type`
- [ ] dedicated SQLite storage for requests that error
- [ ] retrieve() should also be able to return if a value is pending (in addition to present/absent)
- [ ] right now, (for batch) bookkeep_call() is the one that emits an error if a value is pending but not available, but this seems unelegant and should probably be moved out
    

- Automatically persist upon pllm exit

