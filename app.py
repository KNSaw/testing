# llm_cf.py
from dataclasses import dataclass
from typing import Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# Gunakan base model kecil untuk Cloud
BASE_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
LOCAL_ADAPTER_DIR = "./gemma2b_lora_adapter"  # LoRA adapter lokal (jika ada)

@dataclass
class LLMClientHF:
    device: Optional[str] = None

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading base model {BASE_MODEL_NAME} on {self.device}...")

        # Load tokenizer: jika LoRA adapter ada, bisa pakai folder lokal
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_ADAPTER_DIR, use_fast=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)

        # Load base model kecil
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        # Load LoRA adapter jika ada
        try:
            self.model = PeftModel.from_pretrained(base_model, LOCAL_ADAPTER_DIR)
        except Exception:
            self.model = base_model

        # Set pad token jika tidak ada
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if self.device == "cpu":
            self.model.to("cpu")

        # Pipeline text-generation
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
        )

    def ask(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 64,
        temperature: float = 0.0,
    ) -> str:
        full_prompt = f"<<SYS>>\n{system}\n<</SYS>>\n\n{prompt}" if system else prompt
        do_sample = temperature > 0.0

        out = self.generator(
            full_prompt,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            truncation=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gen = out[0]["generated_text"]
        if gen.startswith(full_prompt):
            gen = gen[len(full_prompt):]
        return gen.strip()

    def ask_many(
        self,
        prompts: List[str],
        system: Optional[str] = None,
        max_tokens: int = 64,
        temperature: float = 0.0,
        batch_size: int = 8,
    ) -> List[str]:
        if system:
            prompts = [f"<<SYS>>\n{system}\n<</SYS>>\n\n{p}" for p in prompts]
        do_sample = temperature > 0.0

        outs = self.generator(
            prompts,
            max_new_tokens=max_tokens,
            batch_size=batch_size,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            truncation=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        gens = []
        for full, prompt in zip(outs, prompts):
            txt = full[0]["generated_text"]
            if txt.startswith(prompt):
                txt = txt[len(prompt):]
            gens.append(txt.strip())
        return gens
