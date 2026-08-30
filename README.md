# LangChain Chatbot

A lightweight Streamlit-based chatbot built with LangChain and Python. The project demonstrates a simple prompt-to-LLM workflow and includes both an OpenAI-backed version and a local Ollama-backed version.

## Overview

This project is designed to help you quickly spin up a conversational AI interface in a browser using:

- Python
- Streamlit
- LangChain
- OpenAI models or a local Ollama model
- Optional LangSmith tracking

The app uses a reusable prompt template and a LangChain chain that sends user input to an LLM and returns the generated response.

## Project Structure

```text
LangChain/
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── chatbot/
    ├── app.py
    └── ollama.py
```

### Files

- `chatbot/app.py` — Streamlit chatbot using OpenAI (`ChatOpenAI`)
- `chatbot/ollama.py` — Streamlit/LLM setup using Ollama (`llama2`)
- `requirements.txt` — Python dependencies
- `.env.example` — example environment configuration

## Features

- Simple chat UI with Streamlit
- Prompt templating with LangChain
- OpenAI integration for cloud-hosted LLMs
- Local LLM support via Ollama
- Optional LangSmith tracing
- Easy environment-based configuration

## Prerequisites

Before running the project, make sure you have:

- Python 3.10+ recommended
- `pip` installed
- A working virtual environment tool (`venv` is included in Python)
- An OpenAI API key if using the OpenAI version
- Ollama installed if using the local model version

## Installation

1. Clone or open the project folder.
2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Environment Configuration

Copy the example environment file and add your credentials:

```bash
copy .env.example .env
```

Then update `.env` with your values:

```env
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_TRACKING_V2=true
```

> If you are not using OpenAI or LangSmith, you can leave those values blank or only set the ones you need.

## Running the Application

### Option 1: OpenAI-based chatbot

```bash
streamlit run chatbot/app.py
```

This version uses `ChatOpenAI` with the model `gpt-3.5-turbo`.

### Option 2: Ollama-based chatbot

First, make sure Ollama is installed and the model is available:

```bash
ollama pull llama2
```

Then run:

```bash
streamlit run chatbot/ollama.py
```

This version uses `langchain_community.llms.Ollama` and connects to the local `llama2` model.

## How the App Works

The app builds a prompt template like this:

- System message: instructs the assistant to answer helpfully
- User message: passes the actual question from the input box

The prompt is then combined with the selected LLM and passed through a string output parser.

For the OpenAI version, the chain looks like:

```python
prompt | llm | output_parser
```

This means:

1. The user enters a question
2. The system prompt is combined with the question
3. The model generates a response
4. The output is displayed in the Streamlit interface

## LangSmith / Tracing

The project includes optional LangSmith tracking configuration:

```python
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
```

If you want to enable LangSmith tracing, add your API key in `.env`.

## Customization Ideas

You can extend this project in many ways:

- Add conversation history
- Support multiple models
- Add a sidebar for model selection
- Persist chat sessions
- Enable streaming responses
- Add a more advanced RAG workflow with vector search

## Troubleshooting

### Streamlit not found

```bash
pip install streamlit
```

### OpenAI API key issues

- Make sure `.env` is saved correctly
- Ensure the key is active and valid
- Confirm the environment is loaded before the app starts

### Ollama model not found

```bash
ollama list
ollama pull llama2
```

### Dependency issues

Reinstall dependencies from the requirements file:

```bash
pip install --upgrade -r requirements.txt
```

## License

This project is provided for learning and experimentation. Add your preferred license if you plan to share or distribute it publicly.

## Notes

This project is intentionally minimal and is a good starting point for building more advanced LangChain chat experiences. If you want to develop this into a production-ready chatbot, consider adding:

- session memory
- user auth
- file upload support
- database-backed chat history
- safer prompt engineering and guardrails
