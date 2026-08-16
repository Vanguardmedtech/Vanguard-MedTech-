import streamlit as st
import torch

st.set_page_config(page_title="Vanguard MedTech AI", page_icon="🩺", layout="wide")

# Custom Glassmorphism CSS for Streamlit
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #090614 0%, #120a2a 50%, #0a1128 100%) !important;
        color: #e2e8f0 !important;
    }
    .vanguard-greeting {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #c084fc, #818cf8, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.6rem;
        text-align: center;
    }
    .vanguard-subtext {
        font-size: 1.15rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🏥 Vanguard MedTech")
    st.markdown("---")
    if st.button("➕ New Consultation", use_container_width=True):
        st.session_state.messages = []
    st.markdown("### **Recent Consultations**")
    st.markdown("• *Pediatric Fever Evaluation*")
    st.markdown("• *Malaria Diagnostic Protocol*")
    st.markdown("• *Respiratory Distress Protocol*")
    st.markdown("• *Maternal Care Triage*")

# Hero Header
st.markdown('<div class="vanguard-greeting">Welcome to Vanguard MedTech</div>', unsafe_allow_html=True)
st.markdown('<div class="vanguard-subtext">How can Vanguard AI assist your clinical workflow today?</div>', unsafe_allow_html=True)

# System Prompt
system_prompt = """You are the Vanguard MedTech Clinical Assistant, a specialized AI for healthcare providers across Africa.

When responding to clinical inquiries, ALWAYS format your output clearly using standard text bolding like this:

**1. PRIMARY TRIAGE & URGENCY LEVEL**
- Assess severity (Emergency / Urgent / Routine).

**2. DIFFERENTIAL DIAGNOSES**
- List primary conditions to consider based on symptoms and regional endemicity.

**3. RECOMMENDED CLINICAL PROTOCOL**
- Ground recommendations strictly in WHO or National Standard Treatment Guidelines (STGs).

**4. RED FLAGS & EMERGENCY REFERRAL**
- Highlight immediate warning signs requiring urgent escalation or transfer.

**5. CLINICAL DISCLAIMER**
- State: "Vanguard AI is a decision-support tool. Clinical decisions must be made by a qualified healthcare professional."
"""

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Existing Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ask Vanguard AI..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Format turns for conversation history
    history_formatted = []
    for m in st.session_state.messages[:-1]:
        history_formatted.append({"role": m["role"], "content": m["content"]})

    # Execute Clinical Logic
    user_input_str = prompt
    
    # Check retrieve function if defined in helper scripts
    try:
        clinical_context = retrieve_pan_african_context(user_input_str)
    except NameError:
        clinical_context = None

    if clinical_context:
        augmented_input = f"Relevant Protocol Context:\n{clinical_context}\n\nClinical Inquiry:\n{user_input_str}"
    else:
        augmented_input = user_input_str

    messages = [{"role": "system", "content": system_prompt}] + history_formatted + [{"role": "user", "content": augmented_input}]

    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=350,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

        response = tokenizer.batch_decode(
            [out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)],
            skip_special_tokens=True
        )[0]
    except Exception as e:
        response = f"Clinical Processing Error: {str(e)}. Please retry your prompt."

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
