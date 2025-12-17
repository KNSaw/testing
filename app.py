import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "unsloth/gemma-2-2b-it"
ADAPTER = "diordty/gemma2b-lora-pmb-uajy"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, ADAPTER)
model.eval()

# Test
inputs = tokenizer("Apa syarat pendaftaran S1 UAJY?", return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(out[0], skip_special_tokens=True))
