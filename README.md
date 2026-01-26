# Parallellm

(Parallel Language Models) *p*-LLM



## Compatibility


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

| Function Calls | Sync | Async | Batch |
| --- | --- | --- | --- |
| OpenAI | ✅ | ✅ | ✅ |
| Anthropic | ✅ | ✅ | TODO |
| Google | ✅ | ✅ | ✅ |

| Web Search | Sync | Async | Batch |
| --- | --- | --- | --- |
| OpenAI | ✅ | ✅ | ✅ |
| Anthropic | ✅ | ✅ | TODO |
| Google | ✅ | ✅ | ✅ |



## Philosophy

1. A library designed with the Batch API in mind.
2. We aim to support pipelines where LLMs are "input/output machines", rather than an interactive conversational agent.
3. LLM pipeline control flow should be represented with Python, rather than data structures (ie. LangChain, LangGraph).
4. Circumvent Vendor Lock-in and "Architecture Lock-in".
5. Effortless parallelization.
6. Improved Developer Experience (Develop and debug as synchronous, quickly scale up to huge pipelines).