from datasets import load_dataset

ds = load_dataset("yahma/alpaca-cleaned")["train"]

def get_row(i):
    return {
        "conversations": [
            {
                "role": "user",
                "content": ds[i]["instruction"] + "\n" + ds[i]["input"]
            },
            {
                "role": "assistant",
                "content": ds[i]["output"]
            }
        ]
    }

with open("alpaca_cleaned.jsonl", "w") as f:
    for i in range(len(ds)):
        f.write(f"{get_row(i)}\n")

