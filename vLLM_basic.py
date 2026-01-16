from vllm import LLM, SamplingParams

prompts = [
    "The president of US is ",
    "The colour of apple is ", 
    "Who established Arya Samaj in India? "
]
sampling_params = SamplingParams(temperature = 0.5, top_p = 0.95, max_tokens = 50)
llm = LLM(
    model = "facebook/opt-125m",
    gpu_memory_utilization = 0.4)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    result = output.outputs[0].text
    print(f"Prompt : ", prompt, "Text generated : ", result, "\n")
