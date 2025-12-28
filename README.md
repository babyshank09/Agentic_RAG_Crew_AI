# Agentic RAG Crew AI

A powerful framework combining Retrieval-Augmented Generation (RAG) with multi-agent orchestration using CrewAI.

## Overview

This project demonstrates an intelligent agentic system that leverages RAG capabilities to process, retrieve, and reason over documents while coordinating multiple AI agents to complete complex tasks.

## Features

- **RAG Integration**: Semantic search and document retrieval
- **Multi-Agent System**: Coordinated agents using CrewAI
- **Agentic Workflows**: Autonomous task execution and reasoning

## Getting Started

### Prerequisites

- Python 3.8+
- Git

### Clone the Repository

```bash
git clone https://github.com/babyshank09/Agentic_RAG_Crew_AI.git
cd Agentic_RAG_Crew_AI
```

### Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
python -m streamlit run app.py
```

## Configuration

Set up the serper api key as the environment variable in a `.env` file:
```
SERPER_API_KEY=your_api_key
```

## Project Structure

```
Agentic_RAG_Crew_AI/
├─ README.md
├─ app.py
├─ pyproject.toml
├─ requirements.txt
├─ knowledge/
├─ tests/
└─ src/
    ├─ init.py
    └─ agentic_rag/
        ├─ init.py
        ├─ crew.py # main crew logic
        ├─ main.py # entry point / inputs
        ├─ config/
            ├─ agents.yaml # agent definitions
            └─ tasks.yaml # task definitions
        └─ tools/
            ├─ init.py
            └─ rag_tool.py # RAG tool helper
```
