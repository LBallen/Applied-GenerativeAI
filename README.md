# 🧠 GenAI Tools — A Collection of Applied Generative AI Projects

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-brightgreen?logo=chainlink)
![OpenAI](https://img.shields.io/badge/LLMs-GPT--4o%20%7C%20OpenAI-lightgrey?logo=openai)
![VectorDB](https://img.shields.io/badge/VectorDB-FAISS%20%7C%20Chroma-orange)
![Frameworks](https://img.shields.io/badge/Frameworks-Gradio%20%7C%20Streamlit-pink)
![License](https://img.shields.io/badge/License-MIT-yellow)
![CI](https://img.shields.io/github/actions/workflow/status/<your-username>/genai-tools/ci.yml?label=CI)

A curated repository of **Generative AI applications**, experiments, and prototypes — built to explore how **LLMs**, **retrieval systems**, and **multi-agent pipelines** can be integrated into real-world solutions.  

This monorepo is structured for scalability: each tool (chatbot, agent, or pipeline) lives under `/tools/<tool-name>`, powered by a shared modular foundation in `/common/`.

---

## ⚙️ Current Projects

| Tool                                | Description | Tech Stack                                     |
|-------------------------------------|--------------|------------------------------------------------|
| [`rag-chatbot`](tools/rag-chatbot/) | Retrieval-Augmented Generation chatbot that answers questions based on live web content. | LangChain, FAISS, GPT-4o-mini, Gradio          |
| `Fine-Tuning a LLM`                 | Fine-Tuning RoBERTa on Semantic Similarity (GLUE MRPC) with PEFT/LoRA | Huggingface, Pytorch, Scikit-learn, Matplotlib |
| *(Coming Soon)* `Placeholder`       |  | LangGraph, OpenAI Functions                    |
| *(Coming Soon)* `Placeholder`       |  | LangChain, Chroma, Streamlit                   |

---

## 🧠 Why This Exists

Generative AI tools are powerful — but fragmented.  
This repo brings **practical implementations** of key GenAI patterns into one place for learning, research, and deployment.

Each subproject:
- Demonstrates a **core GenAI concept** (RAG, Agents, Evaluators, etc.)
- Uses **reproducible, clean code**
- Provides **a documented workflow and interface**
- Can run **locally, on Streamlit Cloud, or Hugging Face Spaces**

---

## 🏗️ Repository Structure PLACEHOLDER

```
genai-tools/
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── common/                 # Shared config, logging, prompts
│   ├── config.py
│   ├── logging.py
│   └── prompts/
│       └── example_system_prompt.txt
├── tools/                  # Individual GenAI projects
│   ├── rag-chatbot/
│   │   ├── app.ipynb
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   └── assets/
│   ├── pdf-qa/
│   └── agent-lab/
├── scripts/                # Developer utilities
│   ├── new_tool.sh
│   └── verify_secrets.sh
└── .github/
    ├── workflows/ci.yml
    └── ISSUE_TEMPLATE/
        ├── bug_report.md
        └── feature_request.md
```

---

## 🚀 Quick Start

### 1️⃣ Clone and Setup
```bash
git clone https://github.com/<your-username>/genai-tools.git
cd genai-tools
python3 -m venv .venv
source .venv/bin/activate
```

### 2️⃣ Install Developer Dependencies
```bash
pip install -U pip setuptools wheel pre-commit
pre-commit install
```

### 3️⃣ Configure Environment
Copy the sample environment file and fill in your keys:
```bash
cp .env.example .env
```

**Example:**
```bash
OPENAI_API_KEY="sk-xxxxxx"
USER_AGENT="GenAI-Tools/1.0 (your_email@example.com)"
```

> ⚠️ **Never commit `.env` files or API keys.** Use GitHub Secrets for CI/CD.

---

## ▶️ Running a Tool

Each project has its own README and dependencies.  
To run the **RAG Chatbot**:

```bash
cd tools/rag-chatbot
pip install -r requirements.txt
python app.py
# or run notebook in Jupyter / VSCode
```

Then open the URL shown by Gradio (e.g., http://127.0.0.1:7860).

---

## 🧩 Common Utilities

| Module | Purpose |
|---------|----------|
| `common/config.py` | Loads environment variables (OpenAI API key, user agent, etc.) |
| `common/logging.py` | Provides standardized, colorized logging across tools |
| `common/prompts/` | Stores reusable system/user prompt templates |

---

## 🌱 How to Add a New Tool

You can scaffold a new project automatically:

```bash
./scripts/new_tool.sh <tool-name>
```

This creates:
```
tools/<tool-name>/
├── app.py
├── requirements.txt
└── README.md
```

Then update:
- `README.md` for that tool  
- Add it to the CI matrix under `.github/workflows/ci.yml`  
- Commit and push your new project

---

## 🔬 Example Architecture (RAG Workflow)

```text
User Query
   │
   ▼
[Gradio UI]  →  [Intent Classifier] 
   │
   ▼
[RAG Pipeline]
   ├── WebBaseLoader  →  retrieve URLs
   ├── RecursiveCharacterTextSplitter
   ├── OpenAI Embeddings
   ├── FAISS VectorStore
   └── ChatOpenAI (GPT-4o-mini)
   │
   ▼
[Response → Gradio Chat]
```

---

## 🧠 Planned Additions

- [ ] PDF & document ingestion tools  
- [ ] Multi-agent orchestration with LangGraph  
- [ ] Custom evaluation metrics (faithfulness, retrieval precision)  
- [ ] Dockerfile for deployment  
- [ ] Optional Streamlit frontends  
- [ ] Integration with n8n or FastAPI backends  

---

## 🧪 Development Notes

- Use **Conventional Commits**: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`  
- Keep PRs small and documented.  
- Run `pre-commit run --all-files` before pushing.  
- Do not commit secrets or `.env` files.

---

## 🤝 Contributing

1. Fork the repo  
2. Create a feature branch  
3. Add your tool or improvement  
4. Test and document your changes  
5. Submit a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Leonardo Ballen**  
Mechanical Engineer • AI Developer • RAG Systems Designer  
📧 [lnrdballen@gmail.com](mailto:lnrdballen@gmail.com)  
🔗 [linkedin.com/in/leonardoballen](https://linkedin.com/in/leonardoballen)  
💻 [github.com/<your-username>](https://github.com/<your-username>)

---

## 🏁 Summary

**GenAI Tools** is your growing collection of hands-on **Generative AI projects** — from RAG systems to intelligent agents.  
Each project is a standalone example of practical AI implementation using **LangChain**, **OpenAI APIs**, and **retrieval pipelines**, designed for engineers, students, and innovators building the next wave of AI applications.

---
