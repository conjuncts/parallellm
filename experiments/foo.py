from parallellm.core.gateway import ParalleLLM
# from parallellm.core.manager import BatchManager


print("Before")

pllm = ParalleLLM.resume_directory(".pllm")

with pllm:
    print("This will always be executed")
    pllm.goto_stage("begin")

with pllm:
    pllm.when_stage("begin")

    print("Inside stage 'begin' (the starting stage)")
    resp = pllm.ask_llm(
        "You are a helpful assistant",
        "Please name 8 NFL teams. Place your final answer in a code block, separated by newlines.",
    )

    pllm.goto_stage("end")

with pllm:
    pllm.when_stage("end")

    print("Inside stage 'end' (the terminal stage)")
print("After")

pllm.persist()
