import torch
import transformers
from transformers import AutoTokenizer, OPTForCausalLM
from train import PROMPT_DICT

def predict(payload):
    model = OPTForCausalLM.from_pretrained("facebook/opt-125m")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    source = prompt_input.format_map(payload) if payload.get("input", "") != "" else prompt_no_input.format_map(payload)
    # source = "Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:".format_map(payload)

    # prompt = "Hey, are you consciours? Can you talk to me?"
    inputs = tokenizer(
        source,
        return_tensors="pt",
        padding="longest",
        max_length=1024,
        truncation=True,
    )

    generate_ids = model.generate(inputs.input_ids, max_length=100)
    result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return result


if __name__ == '__main__':
    payload = {
        "instruction": "What is an alpaca?",
        "input": ""
    }
    print(predict(payload))
