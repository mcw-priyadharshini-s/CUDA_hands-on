from vllm import SamplingParams, LLM

prompts = [
    "Give a precise answer for the following question.\nThe president of US is ",
    "Give a precise answer for the following question.\nThe colour of apple is ", 
    "Give a precise answer for the following question.\nWho established Arya Samaj in India? "
]

samplingparams = SamplingParams(temperature = 0.0, top_p = 0.95, max_tokens = 200)

llm = LLM (
    model = "microsoft/Phi-3.5-mini-instruct",
    gpu_memory_utilization = 0.8,
    disable_log_stats = False,
    max_model_len = 2048,
    max_num_seqs = 2,
    dtype = "float16"
)

outputs = llm.generate(prompts, samplingparams)
for out in outputs:
    prompt = out.prompt
    result = out.outputs[0].text
    print("\n\nPrompt: ", prompt, "\nAnswer: ", result)