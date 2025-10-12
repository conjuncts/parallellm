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
- [ ] right now, provider and backend control is inverted. Here, the ask_llm calls provider calls backend. And backend has hard-coded stuff per provider. But really, is should be ask_llm calls backend calls provider. And the provider has encapsulated method that handles input (call_id, llm, etc. --> {sync function, async function}) and post-input format handling (Pydantic Model --> common data format for SQL: this right here is hard coded into guess_schemas()). ask_llm already calls backend anyways to check for cached values. I'm going to pre-empt this by making some of the batch methods have the correct control.
- [ ] dedicated SQLite storage for requests that error
    

- Automatically persist upon pllm exit

