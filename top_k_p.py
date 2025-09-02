from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prompt
prompt = "Write a short poem about the moon landing in 1969:\n"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generation wrapper
def generate_sample(temperature=1.0, top_k=50, top_p=0.95, use_sampling=True):
    gen_args = {
        "input_ids": input_ids,
        "max_new_tokens": 100,
        "pad_token_id": tokenizer.eos_token_id
    }

    if use_sampling:
        gen_args.update({
            "do_sample": True,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p
        })
    else:
        gen_args["do_sample"] = False  # greedy decoding (deterministic)

    output = model.generate(**gen_args)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Run examples
print("🎯 Sampling: temp=0.7, top_k=50, top_p=0.95\n")
print(generate_sample(temperature=0.7, top_k=50, top_p=0.95, use_sampling=True))

print("\n🎯 Deterministic (greedy decoding, no sampling)\n")
print(generate_sample(use_sampling=False))

print("\n🎯 Sampling: temp=1.5, top_k=100, top_p=0.98\n")
print(generate_sample(temperature=1.5, top_k=100, top_p=0.98, use_sampling=True))
