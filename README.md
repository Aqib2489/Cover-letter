# Cover Letter Generator

A simple Streamlit app that generates cover letters using the InternVL3-8B vision-language model.

---

## What It Does

- Takes company name, position, and experience as input
- Generates a tailored cover letter using InternVL3-8B model
- Displays the result in the browser

---

## Requirements

```
streamlit
torch
transformers
```

**Hardware requirements:**
- GPU recommended (model is 8B parameters)
- ~16GB RAM minimum
- CUDA-compatible GPU for faster inference

---

## Setup

1. **Install dependencies:**
```bash
pip install -r requirement.txt
```

2. **Run the app:**
```bash
streamlit run app.py
```

**Note:** First run will download the InternVL3-8B model (~8GB) from HuggingFace.

---

## Usage

1. Enter **Company Name** (e.g., "Google")
2. Enter **Position Title** (e.g., "Software Engineer")
3. Enter **Your Experience Summary** (e.g., "5 years in Python development")
4. Click **Generate Cover Letter**
5. View the generated cover letter

---

## How It Works

```python
# Load InternVL3-8B model (cached after first load)
model, tokenizer = load_model()

# Generate cover letter with prompt
prompt = f"Write a professional cover letter for {position} at {company}..."
response = model.chat(tokenizer, None, prompt, generation_config)
```

**Model:** OpenGVLab/InternVL3-8B  
**Settings:** bfloat16 precision, flash attention, max 1024 tokens

---

## Limitations

- Requires GPU/powerful hardware for decent speed
- Model download needed on first run (~8GB)
- Basic prompt engineering (no advanced customization)
- No history/conversation context
- Generated text quality depends on model capabilities

---

## Author

**Mohammad Aqib**  
Email: maqib@ualberta.ca  
University of Alberta

---
