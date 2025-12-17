import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---------------- CONFIG ----------------
BASE_MODEL = "unsloth/gemma-2-2b-it"
ADAPTER = "./gemma2b_lora_adapter"
MAX_TURNS = 3  # jumlah turn chat yang disertakan di history

# ---------------- LLM Client ----------------
@st.cache_resource(show_spinner=True)
def load_llm():
    st.info("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"  # otomatis ke GPU jika ada, else CPU
    )

    st.info("Applying LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER)
    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    return tokenizer, model

class LLMClient:
    def __init__(self):
        self.tokenizer, self.model = load_llm()

    def ask(self, prompt, max_tokens=128, temperature=0.0):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

# ---------------- Prompt Formatter ----------------
def format_prompt(question, history=""):
    return f"""### Category: PMB_Umum
### Instruction:
Jawablah pertanyaan berikut berdasarkan informasi resmi PMB Universitas Atma Jaya Yogyakarta.

### Input:
{history}
{question}

### Response:
""".strip()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Chatbot PMB UAJY", layout="wide")
st.title("🎓 Chatbot PMB UAJY")
st.caption("Chatbot berbasis LLM Gemma-2-2B + LoRA adapter")

# session state untuk menyimpan chat
if "messages" not in st.session_state:
    st.session_state.messages = []

llm = LLMClient()

# tampilkan riwayat chat
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

# input user
user_input = st.chat_input("Tanyakan seputar PMB UAJY...")

if user_input:
    st.session_state.messages.append(("user", user_input))

    # ambil history terakhir
    history_text = ""
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
