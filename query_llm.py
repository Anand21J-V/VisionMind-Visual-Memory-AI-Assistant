import os
import json
import requests

# === CONFIGURATION ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Export before running: export GROQ_API_KEY=your_key
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_MODEL = "llama3-70b-8192"
MEMORY_FILE = "memory_log.json"

# === PROMPT TEMPLATE (INLINE) ===
PROMPT_TEMPLATE = """
You are a visual memory assistant.

You are given a JSON list of object tracking information from a video. Each object has:
- frame: the frame number it appeared
- object: the class name (e.g. "keys", "remote")
- id: unique ID of the object
- bbox: bounding box coordinates in the video frame

Your task is to answer questions about object locations and activity based on this memory log.

Visual Memory:
{memory}

Now, answer this question:
"{question}"

Only answer what can be inferred from the memory.
If the object isn't in memory, say "I couldn't find that object in the video."
"""

# === Load Memory Log ===
def load_memory(memory_file):
    try:
        with open(memory_file) as f:
            memory = json.load(f)
        return memory
    except FileNotFoundError:
        print(f"❌ Error: {memory_file} not found.")
        return []

# === Ask Question to Groq LLM ===
def ask_question(question, memory_data):
    prompt = PROMPT_TEMPLATE.format(memory=json.dumps(memory_data, indent=2), question=question)

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful visual assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        reply = response.json()["choices"][0]["message"]["content"]
        print(f"\n🧠 Answer: {reply}\n")
    else:
        print(f"❌ Error from Groq: {response.status_code} - {response.text}")

# === MAIN CLI LOOP ===
if __name__ == "__main__":
    memory_data = load_memory(MEMORY_FILE)

    if not memory_data:
        exit(1)

    print("🧠 Visual Memory QA System (Groq + LLaMA 3.3 70B)")
    print("Ask questions like: Where are my keys? What frame had the remote?\n")

    while True:
        user_input = input("❓ Ask a question (or 'exit'): ")
        if user_input.strip().lower() == "exit":
            break
        ask_question(user_input, memory_data)
