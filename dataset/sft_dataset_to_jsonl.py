from datasets import load_dataset
import json

ds = load_dataset("yahma/alpaca-cleaned")["train"]

def get_row(i):
    return json.dumps({
        "conversations": [
            {
                "role": "user",
                "content": f'{ds[i]["instruction"]} {"\n" + ds[i]["input"]}'
            },
            {
                "role": "assistant",
                "content": ds[i]["output"]
            }
        ]
    })

with open("alpaca_cleaned.jsonl", "w") as f:
    for i in range(len(ds)):
        f.write(f"{get_row(i)}\n")

