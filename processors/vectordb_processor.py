__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import importlib
import traceback
import chromadb
import ocr_processor
import splitter_processor
import os
from pathlib import Path
import shutil
from langchain.chains import RetrievalQA 
from langchain_core.documents import Document
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.llms.vllm import VLLMOpenAI
from langchain_chroma import Chroma
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.structured_query import StructuredQuery
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from chromadb.config import Settings
from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv
import us_states
importlib.reload(ocr_processor)
importlib.reload(splitter_processor)
importlib.reload(us_states)
load_dotenv()

DEFAULT_LANGCHAIN_COLLECTION_NAME='langchain'

class VectorDbProcessor:
    def __init__(self, llm: str, embed_model: str, **kwargs):
        """
        Initialize relevant settings for LLMs, OCR and Splitter.
        """
        print("Initializing OCR Processor...")
        self.ocr = ocr_processor.OcrProcessor()

        print("Initializing splitter...")
        self.splitter = splitter_processor.SplitterProcessor()
        
        print("Initializing Vector DB/LLM settings...")
        self.initialize_db_settings(llm, embed_model, **kwargs)
            
    def initialize_db_settings(self, llm: str, embed_model: str, **kwargs):
        print("Initializing settings for LLM model...")
        self.llm = ChatOpenAI(
            model=llm, 
            temperature=0.1,
            api_key=os.getenv('GRANITE_API_KEY'),
            base_url=os.getenv('GRANITE_API_BASE'),
        )
            
        print("Initializing embedding model...")
        self.embed_model = OpenAIEmbeddings(
            api_key=os.getenv('EMBED_API_KEY'),
            base_url=os.getenv('EMBED_API_BASE'),
            dimensions=768,
            model=embed_model,
        )
        
        print(f"Initializing ChromaDb Client...")
        self.chroma_client = chromadb.HttpClient(host= f"http://{os.getenv('CHROMA_API_BASE')}", settings=Settings(allow_reset=True))
        # self.chroma_client = chromadb.PersistentClient(path=f"{Path.cwd()}/db")
            
        print(f"Initializing Vector Store...")
        self.vector_store = Chroma(
            client=self.chroma_client,
            embedding_function=self.embed_model,
            persist_directory='/data',
            collection_name=kwargs.get('collection_name') or 'langchain',
        )        

    def load_documents(self, source_dir: str, collection_name: str):
        
        print(f"Creating collection {collection_name} (if it does not exist)...")
        
        chroma_collection = self.chroma_client.get_or_create_collection(collection_name)

        print(f"Loading documents from {source_dir} into collection {collection_name}...")
        
        try:

            for root, _, source_files in os.walk(source_dir):
                
                for source_file in source_files:
                
                    chunks = self.splitter.process(os.path.join(root,source_file))
                    
                    documents = [Document(id=str(uuid4()), 
                                          page_content=chunk, 
                                          metadata={"source_file": source_file, "state": us_states.find_state_in_text(source_file)}) 
                                 for chunk in chunks]
                    
                    self.vector_store.add_documents(documents=documents)
                
            print("Loading complete.")
        except Exception as e: 
            print(f"Error loading docs: {e}") 
            
            traceback.print_exc()

    def process(self, prompt_input: str, system_prompt_input:str = None) -> str:
        try:
            system_prompt_text = system_prompt or """Answer any use questions based solely on the context below:
            <context>{context}</context>
            """
            
            user_prompt_text = """{input}"""
    
            prompt = ChatPromptTemplate([
                ("system", system_prompt_text),
                ("human", user_prompt_text),
            ])
            
            combine_docs_chain = create_stuff_documents_chain(self.llm, prompt)
            
            qa_chain = create_retrieval_chain(self.vector_store.as_retriever(), combine_docs_chain)
    
            return qa_chain.invoke({"input": prompt_input})
        except Exception as e: 
            print(f"Error loading docs: {e}") 
            
            traceback.print_exc()

if __name__ == "__main__":  
    
    source_dir = f"{os.path.expanduser('~')}/{os.getenv('APP_NAME')}/scraped/studentaid"
    
    processor = VectorDbProcessor(llm='granite-3-3-8b-instruct',
                                  embed_model='nomic-embed-text-v1.5',
                                  collection_name='scholarships',)
    
    # processor.load_documents(source_dir=source_dir, collection_name=collection_name) # Loads documents into db index

    print(processor.process(prompt_input="What kinds of scholarships are available for veterans in Kentucky?"))