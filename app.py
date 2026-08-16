import gradio as gr

def respond(message, history):
    # Process user input and media attachments
    if isinstance(message, dict):
        text_content = message.get("text", "")
        files = message.get("files", [])
    else:
        text_content = str(message)
        files = []

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

    messages = [{"role": "system", "content": system_prompt}]

    # Process multi-turn chat history
    for turn in history:
        if isinstance(turn, (list, tuple)):
            u_text = turn[0] if len(turn) > 0 and turn[0] else ""
            a_text = turn[1] if len(turn) > 1 and turn[1] else ""
        elif isinstance(turn, dict):
            u_text = turn.get("user", "") or turn.get("content", "")
            a_text = turn.get("assistant", "")
        else:
            u_text, a_text = "", ""

        if u_text:
            messages.append({"role": "user", "content": str(u_text)})
        if a_text:
            messages.append({"role": "assistant", "content": str(a_text)})

    user_input_str = text_content
    if files:
        file_list = ", ".join([str(f) for f in files])
        user_input_str += f"\n[Attached file(s): {file_list}]"

    # Fetch Real-Time & Local Pan-African Clinical Context
    clinical_context = retrieve_pan_african_context(user_input_str)

    if clinical_context:
        augmented_input = f"Relevant Protocol Context:\n{clinical_context}\n\nClinical Inquiry:\n{user_input_str}"
    else:
        augmented_input = user_input_str

    messages.append({"role": "user", "content": augmented_input})

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

    return response

# Custom Glassmorphism CSS
vanguard_css = """
footer, .gradio-container footer, .api-docs, #settings-btn, .built-with,
[data-testid="block-label"], .label, .component-title, .chatbot-header,
.chatbot-label, span.label {
    display: none !important;
}

body, .gradio-container {
    background: linear-gradient(135deg, #090614 0%, #120a2a 50%, #0a1128 100%) !important;
    color: #e2e8f0 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

.vanguard-hero-container {
    text-align: center;
    padding: 35px 15px 15px 15px;
    max-width: 800px;
    margin: 0 auto;
}

.vanguard-greeting {
    font-size: 2.3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #c084fc, #818cf8, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.6rem;
    line-height: 1.2;
}

.vanguard-subtext {
    font-size: 1.15rem;
    color: #94a3b8;
    line-height: 1.5;
}

.message.user {
    background: rgba(124, 58, 237, 0.45) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(168, 85, 247, 0.5) !important;
    color: #ffffff !important;
    border-radius: 20px 20px 4px 20px !important;
    box-shadow: 0 8px 32px 0 rgba(124, 58, 237, 0.25) !important;
}

.message.bot {
    background: rgba(18, 14, 38, 0.65) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    color: #f1f5f9 !important;
    border-radius: 20px 20px 20px 4px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
}

.gradio-container textarea, .gradio-container .multimodal-textbox {
    border-radius: 30px !important;
    background: rgba(20, 15, 40, 0.7) !important;
    backdrop-filter: blur(10px) !important;
    border: 1.5px solid rgba(139, 92, 246, 0.6) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 20px rgba(124, 58, 237, 0.15) !important;
}
"""

custom_theme = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="violet",
    neutral_hue="slate"
)

custom_textbox = gr.MultimodalTextbox(
    interactive=True,
    file_types=["image", "audio"],
    placeholder="Ask Vanguard AI...",
    show_label=False,
    sources=["upload", "microphone"],
    file_count="multiple"
)

custom_chatbot = gr.Chatbot(
    show_label=False,
    avatar_images=None
)

with gr.Blocks(theme=custom_theme, css=vanguard_css, title="Vanguard MedTech AI") as demo:

    with gr.Sidebar(position="left", open=False):
        gr.Markdown("## 🏥 Vanguard MedTech")
        gr.Markdown("---")
        gr.Button("➕ New Consultation", variant="primary")
        gr.Markdown("### **Recent Consultations**")
        gr.Markdown("• *Pediatric Fever Evaluation*")
        gr.Markdown("• *Malaria Diagnostic Protocol*")
        gr.Markdown("• *Respiratory Distress Protocol*")
        gr.Markdown("• *Maternal Care Triage*")

    gr.HTML("""
        <div class="vanguard-hero-container">
            <div class="vanguard-greeting">Welcome to Vanguard MedTech</div>
            <div class="vanguard-subtext">How can Vanguard AI assist your clinical workflow today?</div>
        </div>
    """)

    gr.ChatInterface(
        fn=respond,
        multimodal=True,
        textbox=custom_textbox,
        chatbot=custom_chatbot,
        title="",
        description=""
    )

demo.launch(share=True)
