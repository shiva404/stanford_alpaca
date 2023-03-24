import torch
import transformers
from transformers import AutoTokenizer, OPTForCausalLM
from train import PROMPT_DICT
import random
seed = random.random()
random.seed(int(seed))
torch.cuda.manual_seed(int(seed))

MODEL_DIR="/home/shiv/ml/stanford_alpaca/data/model"
MODEL="alpaca_OPT_125m"
def predict(payload):

    model = OPTForCausalLM.from_pretrained(f"{MODEL_DIR}/alpaca_OPT_125m")
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_DIR}/alpaca_OPT_125m")

    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    source = prompt_input.format_map(payload) if payload.get("input", "") != "" else prompt_no_input.format_map(payload)
    inputs = tokenizer(
        source,
        return_tensors="pt",
        padding="longest",
        max_length=1024,
        truncation=True,
    )

    generate_ids = model.generate(inputs.input_ids, max_length=200)
    result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return result


if __name__ == '__main__':
    payload = {
        "instruction": "Write bed time  story on galaxy",
        "input": ""
    }
    print(predict(payload))
