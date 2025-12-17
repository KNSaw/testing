import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER = "./gemma2b_lora_adapter"

class LLMClient:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.float16,
            device_map="auto"
        )

        self.model = PeftModel.from_pretrained(base_model, ADAPTER)
        self.model.eval()

    def ask(self, prompt, max_tokens=128, temperature=0.0):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Chatbot PMB UAJY")

llm = LLMClient()

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
st.caption("Chatbot berbasis LLM dengan LoRA adapter")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan riwayat chat
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

user_input = st.chat_input("Tanyakan seputar PMB UAJY...")

if user_input:
    st.session_state.messages.append(("user", user_input))

    # Ambil history terakhir
    history_text = ""
    MAX_TURNS = 3
    for role, msg in st.session_state.messages[-2*MAX_TURNS:-1]:
        history_text += f"{role.capitalize()}: {msg}\n"

    prompt = format_prompt(user_input, history_text)

    try:
        with st.spinner("Sedang menjawab..."):
            response = llm.ask(prompt)
    except Exception as e:
        response = f"Terjadi error: {e}"

    st.session_state.messages.append(("assistant", response))

    with st.chat_message("assistant"):
        st.markdown(response)
