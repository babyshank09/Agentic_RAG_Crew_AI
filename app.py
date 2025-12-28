import streamlit as st
import yaml
from crewai import Agent, Crew, Process, Task, LLM
from crewai_tools import SerperDevTool
from src.agentic_rag.tools.rag_tool import DocumentSearchTool

st.set_page_config(page_title="Agentic RAG CrewAI", page_icon="🤖")
st.title("🤖 Agentic RAG CrewAI")
st.markdown("Ask questions and get answers based on the uploaded PDF and web search.")

with st.sidebar:
    st.header("Settings")

    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.warning("Please enter your OpenAI API Key to proceed.")
        st.stop()
    st.session_state.openai_key = openai_api_key

    pdf_file = st.file_uploader("Upload PDF Document", type=["pdf"])
    if not pdf_file:
        st.warning("Please upload a PDF document to proceed.")
        st.stop()

    with open("knowledge/uploaded_file.pdf", "wb") as f:
        f.write(pdf_file.getvalue())

    if "pdf_tool" not in st.session_state:
        st.session_state.pdf_tool = DocumentSearchTool(
            file_path="knowledge/uploaded_file.pdf", 
            openai_api_key=st.session_state.openai_key
        )

def load_llm():
    return LLM(
        model="gpt-4o-mini",
        api_key=st.session_state.openai_key
    )

def create_crew(pdf_tool):
    web_search_tool = SerperDevTool()

    with open("src/agentic_rag/config/agents.yaml", "r") as f:
        agents_config = yaml.safe_load(f)

    with open("src/agentic_rag/config/tasks.yaml", "r") as f:
        tasks_config = yaml.safe_load(f)

    llm = load_llm()

    retriever_agent = Agent(
        role=agents_config["retriever_agent"]["role"],
        goal=agents_config["retriever_agent"]["goal"],
        backstory=agents_config["retriever_agent"]["backstory"],
        tools=[pdf_tool, web_search_tool],
        llm=llm,
        verbose=True
    )

    response_agent = Agent(
        role=agents_config["response_synthesizer_agent"]["role"],
        goal=agents_config["response_synthesizer_agent"]["goal"],
        backstory=agents_config["response_synthesizer_agent"]["backstory"],
        llm=llm,
        verbose=True
    )

    retrieval_task = Task(
        description=tasks_config["retrieval_task"]["description"],
        expected_output=tasks_config["retrieval_task"]["expected_output"],
        agent=retriever_agent
    )

    response_task = Task(
        description=tasks_config["response_task"]["description"],
        expected_output=tasks_config["response_task"]["expected_output"],
        agent=response_agent
    )

    return Crew(
        agents=[retriever_agent, response_agent],
        tasks=[retrieval_task, response_task],
        process=Process.sequential,
        verbose=True
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "crew" not in st.session_state:
    st.session_state.crew = create_crew(st.session_state.pdf_tool)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask me anything...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.crew.kickoff(
                inputs={"query": prompt}
            ).raw
            st.markdown(result)

    st.session_state.messages.append(
        {"role": "assistant", "content": result}
    )
