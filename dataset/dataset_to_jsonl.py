import os
import time
from datasets import load_dataset
import logging

NUM_CPUS = max(1, os.cpu_count() - 2)

DATASET_NAME = "HuggingFaceTB/cosmopedia"
CONFIG_NAME = "stories"
SPLIT = "train[:10%]"
OUTPUT_FILENAME = "cosmopedia_stories.jsonl"
COLUMN_TO_EXPORT = "text"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(f"Optimizing for {NUM_CPUS} vCPUs.")
start_time = time.time()

print(f"Loading dataset '{DATASET_NAME}' ({CONFIG_NAME}/{SPLIT}) using {NUM_CPUS} processes...")
try:
    ds = load_dataset(
        DATASET_NAME,
        CONFIG_NAME,
        split=SPLIT,
        num_proc=NUM_CPUS,
    )
    load_time = time.time()
    print(f"Dataset loaded with {len(ds)} examples in {load_time - start_time:.2f} seconds.")

    if COLUMN_TO_EXPORT not in ds.column_names:
         raise ValueError(f"Error: Column '{COLUMN_TO_EXPORT}' not found in the dataset.")

    columns_to_remove = [col for col in ds.column_names if col != COLUMN_TO_EXPORT]
    if columns_to_remove:
        print(f"Removing columns: {columns_to_remove}")
        # remove_columns is generally fast as it manipulates metadata
        ds = ds.remove_columns(columns_to_remove)
    else:
        print(f"Dataset already contains only the '{COLUMN_TO_EXPORT}' column.")

    prepare_time = time.time()
    print(f"Data preparation finished in {prepare_time - load_time:.2f} seconds.")

    print(f"Writing '{COLUMN_TO_EXPORT}' column to {OUTPUT_FILENAME}...")
    ds.to_json(
        OUTPUT_FILENAME,
        orient='records',
        lines=True,
        force_ascii=False
    )

    write_time = time.time()
    print(f"\nDataset successfully converted to {OUTPUT_FILENAME} in {write_time - prepare_time:.2f} seconds.")
    print(f"Total time: {write_time - start_time:.2f} seconds.")

except Exception as e:
    logger.error(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
