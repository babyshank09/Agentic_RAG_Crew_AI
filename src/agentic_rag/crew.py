from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from src.agentic_rag.tools.rag_tool import DocumentSearchTool

pdf_tool = DocumentSearchTool(file_path='knowledge/sample.pdf')
web_search_tool = SerperDevTool()

@CrewBase
class AgenticRag(): 
    """Agentic RAG Crew""" 

    agents_config = "config/agents.yaml"
    tools_config = "config/tasks.yaml"
 
    @agent 
    def retriever_agent(self) -> Agent:  
        return Agent(
            config = self.agents_config["retriever_agent"],
            verbose=True,
			tools=[
				pdf_tool,
				web_search_tool
			]
        )
    
    @agent 
    def response_synthesizer_agent(self) -> Agent: 
        return Agent(
            config = self.agents_config["response_synthesizer_agent"],
            verbose = True
        )
    
    @task 
    def retrieve_documents(self) -> Task:
        return Task(
            config = self.tasks_config["retrieval_task"],
        )
    
    @task 
    def synthesize_response(self) -> Task:
        return Task(
            config = self.tasks_config["response_task"],
        ) 
    
    @crew
    def crew(self)-> Crew: 
        """Creates the AgenticRag crew"""
        
        return Crew(
            agents = self.agents, 
            tasks = self.tasks,
            process = Process.sequential, 
            verbose = True
        )
    
    


