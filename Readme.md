# 🧠 AI Persona – Chat with Hitesh Choudhary (AI Clone)

An interactive web-based chatbot that mimics the **personality, tone, and storytelling style** of a real person. In this project, I've cloned the essence of **Hitesh Choudhary**, but you can swap the persona for **any creator, founder, or even yourself!**

## 🧰 Tools You'll Need

- **🧠 OpenAI API** – For generating responses using GPT models  
- **🌐 Streamlit** – For building a fast, clean web UI  
- **🔐 python-dotenv** – For loading environment variables securely  
- **💻 Python (>=3.8)** – Your core programming language

## 📁 Project Structure

```
ai-persona/
├── persona.py                    # Main Streamlit app
├── .env                         # Store your OpenAI API key
├── prompts/
│   └── hitesh_examples.json     # ~40 examples of how Hitesh speaks
├── requirements.txt
└── README.md
```

## 🧱 Step-by-Step Build

### 1️⃣ Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2️⃣ Install Dependencies

```bash
pip install streamlit openai python-dotenv
```

### 3️⃣ Add Your API Key

Create a `.env` file:

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 4️⃣ Prompt Engineering is the 💖

To simulate Hitesh Choudhary's unique style:

- Transcripts from 2-3 YouTube videos
- Insights from LinkedIn, Udemy courses
- Tone: Hinglish, emotional, motivational, storytelling

Example prompt:

```json
{
  "role": "user",
  "content": "Hitesh bhai, mujhe backend seekhna hai par kahan se start karu?"
},
{
  "role": "assistant",
  "content": "Bilkul, backend seekhna kaafi exciting journey hai. Dekho sabse pehle Node.js ya Python choose karo..."
}
```

### 5️⃣ Main Streamlit App (persona.py)

```python
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()
client = OpenAI()

SYSTEM_PROMPT = """
You are an AI version of Hitesh Choudhary.
You speak in Hinglish with storytelling flair, use simple analogies, 
emotional motivation, and break down tech concepts for learners. 
You are informal, encouraging, and sound like a relatable mentor.
"""

st.title("🤖 Chat with Hitesh Choudhary (AI)")

query = st.text_input("Aapka sawal?")

if query:
    with open("prompts/hitesh_examples.json", "r") as f:
        examples = json.load(f)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ] + examples + [
        {"role": "user", "content": query}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    
    st.write(response.choices[0].message.content)
```

### 6️⃣ Run the App 🚀

```bash
streamlit run persona.py
```

Visit: `http://localhost:8501`

**Boom 💥** — your own Hitesh AI clone is live!

## 🧠 Modify It For Yourself

Want to create your own AI Persona?

Just update:
- The `SYSTEM_PROMPT` with your tone
- Feed in 10–50 prompt-response pairs matching your style

Example prompt change:

```python
SYSTEM_PROMPT = "You are AI version of [Your Name]. You use Gen Z slang, emojis, and joke a lot."
```

## 🧪 Test the Live Demo

🔗 [Try the AI Persona Demo](https://perosna-ai-vwfxavyehtbnwuwt53rywj.streamlit.app/)

## 🙌 Final Thoughts

- You don't need to host or train a full LLM to build a custom AI
- Just prompt smartly + use OpenAI + Streamlit
- Whether it's your mentor, a favorite creator, or your own clone — the possibilities are endless

🔥 **If you build your own, tag me on X (Twitter) with #AIPersona – I'd love to see it!**

Happy building! 🚀

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](../../issues).

## 📞 Support

If you have any questions or need help, please open an issue or reach out on Twitter.

---

**Made with ❤️ and AI**