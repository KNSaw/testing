import streamlit as st
from dataclasses import dataclass
from typing import Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

LOCAL_ADAPTER_DIR = "./gemma2b_lora_adapter"
BASE_MODEL_NAME = "google/gemma-2b"


@dataclass
class LLMClientHF:
    device: Optional[str] = None

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading base model {BASE_MODEL_NAME} on {self.device}...")
        print(f"Loading LoRA adapter from {LOCAL_ADAPTER_DIR}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_ADAPTER_DIR,
            use_fast=True
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            LOCAL_ADAPTER_DIR,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if self.device == "cpu":
            self.model.to("cpu")

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
        max_tokens: int = 16,
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
        max_tokens: int = 16,
        temperature: float = 0.0,
        batch_size: int = 16,
    ):
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


# ----------------- STREAMLIT UI -----------------

st.set_page_config(page_title="Chatbot PMB UAJY")

@st.cache_resource
def load_llm():
    return LLMClientHF()

llm = load_llm()


def format_prompt(question, history=""):
    return f"""### Category: PMB_Umum
### Instruction:
Jawablah pertanyaan berikut berdasarkan informasi resmi PMB Universitas Atma Jaya Yogyakarta.

### Input:
{history}
{question}

### Response:
""".strip()


st.title("🎓 Chatbot PMB UAJY")
st.caption("Chatbot berbasis LLM hasil fine-tuning")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan history chat
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

user_input = st.chat_input("Tanyakan seputar PMB UAJY...")

if user_input:
    st.session_state.messages.append(("user", user_input))

    history_text = ""
    MAX_TURNS = 3
    for role, msg in st.session_state.messages[-2 * MAX_TURNS:-1]:
        history_text += f"{role.capitalize()}: {msg}\n"

    prompt = format_prompt(user_input, history_text)

    try:
        with st.spinner("Sedang menjawab..."):
            response = llm.ask(prompt, temperature=0.0, max_tokens=128)
    except Exception as e:
        st.error(f"Terjadi error: {e}")
        response = "Maaf, sistem sedang mengalami gangguan."

    st.session_state.messages.append(("assistant", response))

    with st.chat_message("assistant"):
        st.markdown(response)