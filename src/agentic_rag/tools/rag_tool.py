import os
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field, ConfigDict
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings  
from dotenv import load_dotenv 

load_dotenv() 

class DocumentSearchToolInput(BaseModel): 
    """Input schema for DocumentSearchTool."""
    query: str = Field(..., description="The search query to find relevant documents.") 

class DocumentSearchTool(BaseTool):
    name: str = "DocumentSearchTool"
    description: str = "Search the PDF document for the given query string." 
    args_schema: Type[BaseModel] = DocumentSearchToolInput 

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)  

    def __init__(self, file_path: str, openai_api_key: str, embedding_model=None): 
        """Initialize the searcher with a PDF file path and set up ChromaDB."""
        super().__init__()   
        self.file_path = file_path 

        if embedding_model: 
            self.embedding_model = embedding_model 
        else: 
            self.embedding_model = OpenAIEmbeddings(openai_api_key= openai_api_key) 
        
        self._process_document() 

    def _load_pdf(self, file_path: str):  
        """Loads the PDF file and returns the documents.""" 
        loader = PyPDFLoader(file_path) 
        documents = loader.load()
        return documents 
    
    def _create_chunks(self, documents):
        """Splits documents into smaller chunks"""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50) 
        chunks = text_splitter.split_documents(documents) 
        return chunks   
    
    def _process_document(self): 
        """Processes the PDF document and sets up ChromaDB."""
        documents = self._load_pdf(self.file_path) 
        chunks = self._create_chunks(documents)    

        self.vectorstore = Chroma.from_documents( 
            documents=chunks, 
            embedding=self.embedding_model, 
        ) 

        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}) 

    def _run(self, query: str) -> str: 
        """Executes the search for the given query and returns relevant document contents."""
        relevant_documents = self.retriever.invoke(query) 
        docs = [doc.page_content for doc in relevant_documents] 
        return "\n\n".join(docs)



    




