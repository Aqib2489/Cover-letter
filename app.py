import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer

@st.cache_resource(show_spinner=False)
def load_model():
    path = 'OpenGVLab/InternVL3-8B'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto"  # Or your custom device_map
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    return model, tokenizer


    
def generate_cover_letter(model, tokenizer, company, position, experience):
    prompt = (
        f"Write a professional cover letter for the position of {position} at {company}. "
        f"Mention that I have relevant experience in {experience}. "
        f"The tone should be formal and tailored to the role."
    )
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    
    # If you're not using history:
    response = model.chat(tokenizer, None, prompt, generation_config)
    
    return response

def main():
    st.title("InternVL Cover Letter Generator")

    st.markdown("Enter the details below to generate a professional cover letter.")

    company = st.text_input("Company Name")
    position = st.text_input("Position Title")
    experience = st.text_area("Your Experience Summary")

    if st.button("Generate Cover Letter"):
        if not (company and position and experience):
            st.warning("Please fill in all fields.")
            return

        with st.spinner("Generating cover letter..."):
            model, tokenizer = load_model()
            cover_letter = generate_cover_letter(model, tokenizer, company, position, experience)

        st.subheader("Generated Cover Letter")
        st.write(cover_letter)

if __name__ == "__main__":
    main()
