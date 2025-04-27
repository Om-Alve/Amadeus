import sys
import os
import torch
import warnings
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model import AmadeusConfig, AmadeusForCausalLM 

warnings.filterwarnings('ignore')

MODEL_CHECKPOINT_PATH = 'out/pretrain_512.pth'
TOKENIZER_PATH = './model/'
HIDDEN_SIZE = 512         
NUM_HIDDEN_LAYERS = 8    
USE_MOE = False         
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INPUT_PROMPT = "Once upon a time, there was a cat"
MAX_NEW_TOKENS = 100   
TEMPERATURE = 0.85    
TOP_P = 0.85         

# --- Load Tokenizer ---
print(f"Loading tokenizer from: {TOKENIZER_PATH}")
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    print("Please ensure the tokenizer files are present in the './model/' directory.")
    exit()

print("Initializing model configuration...")
config = AmadeusConfig(
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=NUM_HIDDEN_LAYERS,
)

print("Initializing model structure...")
model = AmadeusForCausalLM(config)

print(f"Loading model weights from: {MODEL_CHECKPOINT_PATH}")
try:
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE), strict=True)
    print("Model weights loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model checkpoint not found at {MODEL_CHECKPOINT_PATH}")
    exit()
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit()

model.eval()  
model.to(DEVICE)
print(f"Model moved to device: {DEVICE}")
print(f'Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')


input_text = tokenizer.bos_token + INPUT_PROMPT if tokenizer.bos_token else INPUT_PROMPT
print(f"Input text (with BOS if applicable): {input_text!r}")

inputs = tokenizer(
    input_text,
    return_tensors="pt", 
    truncation=False 
).to(DEVICE)

print(f"Input IDs shape: {inputs['input_ids'].shape}")

print("\nGenerating text...")
output_text = ""
try:
    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=MAX_NEW_TOKENS,
            num_return_sequences=1,
            do_sample=True, 
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id, 
            eos_token_id=tokenizer.eos_token_id,
            top_p=TOP_P,
            temperature=TEMPERATURE
        )

    output_ids = generated_ids[0][inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    print("Generation complete.")

except Exception as e:
    print(f"Error during text generation: {e}")

print("\n--- Generation Result ---")
print(f"Prompt: {INPUT_PROMPT}")
print(f"Generated Continuation: {output_text}")
print("-------------------------\n")
