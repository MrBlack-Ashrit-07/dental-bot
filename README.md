# 🦷 Dental Clinic RAG Assistant

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Deployment](https://img.shields.io/badge/deployment-Render-purple)

> An intelligent, Retrieval-Augmented Generation (RAG) chatbot designed to automate patient inquiries, explain dental procedures, and streamline clinic communications. 

**[🔴 Try the Live Demo Here](https://dental-bot-n10s.onrender.com)**

---

## 🚀 Overview
<img width="676" height="863" alt="Screenshot (72)" src="https://github.com/user-attachments/assets/d45cecae-a5a0-4028-9828-d506c837c92c" />
Managing patient questions about treatments, post-care instructions, and clinic policies takes up hours of administrative time. This bot ingests specific dental documentation and clinic guidelines to provide patients with instant, accurate, and context-aware answers 24/7.

## 🏗️ Architecture & Tech Stack
* **Framework:** Flask
* **LLM:** Groq (Llama 3.3 70B)
* **Vector Database:** TF-IDF via scikit-learn (swapped from ChromaDB due to memory constraints
* **Deployment:** Render

## ⚙️ Core Capabilities
* **Context-Aware Responses:** Grounded entirely in provided dental literature to prevent hallucinations on sensitive medical topics.
* **Procedure Explanations:** Breaks down complex treatments (e.g., root canals, implants) into patient-friendly language.
* **Post-Op Care:** Retrieves specific aftercare instructions based on the procedure mentioned.

## 💻 Local Setup
To run this clinic bot locally:

```bash
# Clone the repo
git clone [https://github.com/yourusername/dental-clinic-bot.git](https://github.com/yourusername/dental-clinic-bot.git)

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (add your API keys here)
cp .env.example .env

# Run the application
python app.py
