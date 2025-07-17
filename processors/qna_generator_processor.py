from localtemplates.prompts import MAIN_PROMPT
# from vllm import LLM, SamplingParams
# import ollama
from langchain_openai import ChatOpenAI
from langchain.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import random
import os
import glob
import traceback
import splitter_processor
import ocr_processor
import qna_cleanup_processor
import json
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString
from io import StringIO
from pathlib import Path
import traceback
import textwrap
from dotenv import load_dotenv
load_dotenv()
import importlib
importlib.reload(splitter_processor)
importlib.reload(ocr_processor)

class QnaGeneratorProcessor:
    def __init__(self, model_id: str):
        """
        Initialize the QnaGeneratorProcessor.
        """
        self.llm = ChatOpenAI(model=model_id, 
                              temperature=0
                             )
        
        self.num_contexts = 12 # Currently supports 12 contexts per qna.yaml file
        
        self.num_contexts_per_file = [12] # This will select up to 8 files (12 contexts total)
        
        self.splitter = splitter_processor.SplitterProcessor()

        self.ocr = ocr_processor.OcrProcessor()
        
        self.refiner = qna_cleanup_processor.QnaCleanupProcessor()

        self.model_id = model_id
        
    def select_chunks(self, input_dir: str, table_dir: str) -> list:
        """
        Selects a subset of files from input_dir and generates contexts for each file as a list of strings.
        """
        print(f"\n\nApplying semantic chunking to directory {input_dir}...")
        input_files = glob.glob(f"{input_dir}/*.md")
        table_files = glob.glob(f"{table_dir}/*.md") if table_dir else []

        if not input_files:
            print(f"No Markdown files found in <{input_dir}>.")
            return

        input_files.sort(key=lambda item: os.path.getsize(item), reverse=True)
        
        sample_size = len(self.num_contexts_per_file)
        
        context_counts = [cnt for cnt in self.num_contexts_per_file]
        
        random.shuffle(context_counts)
        
        print(f"Context counts: {context_counts}")
        
        chunks = []
        
        for i, count in enumerate(context_counts):
            new_chunks = self.splitter.process(input_files[i])
            chunks += new_chunks[:count]

        # Add at least 1 table context if one exists
        if len(table_files):
            print(f"Adding table context from {table_files[0]}...")
            chunks[-1] = self.ocr.extract_tables(table_files[0])[0] 

        print(f"Semantic chunking completed.")
        
        return chunks

    def generate_question_answer_pairs(self, chunk: str) -> list:
        """
        From the given chunk, uses model <model_id> to generate a list of question answer pairs.
        """

        prompt = PromptTemplate.from_template(MAIN_PROMPT)

        response = ( prompt | self.llm).invoke({"context": chunk})

        return response.content

    def generate_yaml_payload(self, chunks) -> dict:
        print("\n\nGenerating yaml content...")
        payload = {"version": 3,
                   "domain" : "",
                   "created_by": "",
                   "seed_examples": [],
                   "document_outline": "",
                   "document": {
                       "repo": "",
                       "commit": "",
                       "patterns": [""],
                   }
                  }
        
        for i, chunk in enumerate(chunks):
            try:
                literal_chunk = LiteralScalarString(f"""{'\n'.join(textwrap.wrap(chunk.rstrip(),70))}""")
                section = {"context": literal_chunk,
                           "questions_and_answers": json.loads(self.generate_question_answer_pairs(chunk))}
                
                payload["seed_examples"].append(section)
            except Exception as e:
                print(f"Error while processing chunk {i}...")
                traceback.print_exc()
                print("Skipped chunk.")

        print(f"Generated YAML payload from template.")
        
        return payload
    
    def generate_yaml_file(self, payload: dict, output_dir: str) -> None:
        
        print("\n\nGenerating yaml file...")
        with open(f"{output_dir}/qna.yaml", "w") as f:
            yaml = YAML()
            yaml.indent(mapping=2, sequence=4, offset=2)
            yaml.default_flow_style = False
            yaml.default_style=None
            yaml.preserve_quotes=False
            yaml.allow_unicode = True
            yaml.encoding = 'utf-8'
            yaml.explicit_start = True

            yaml.dump(payload, f, transform=self.refiner.process_chunk )
            
            print(f"qna.yaml file generated at {output_dir}/qna.yaml.")

    def process(self, input_dir: str, output_dir: str, table_dir=None) -> None:
        """
        Splits docs using semantic chunking.

        Args:
            input_dir (str): Directory containing converted markdown files
            output_dir (str): Target directory for qna.yaml file
            table_dir (str): Directory containing converted markdown files with tables
        """
        input_files = glob.glob(f"{input_dir}/*.md")

        if not input_files:
            print(f"No Markdown files found in <{input_dir}>.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
            
        with open(f"{output_dir}/qna.yaml", "w") as f:
            print(f"Generating file: {f.name}...")
            
            chunks = self.select_chunks(input_dir, table_dir)

            payload = self.generate_yaml_payload(chunks)

            print(f"YAML: \n{json.dumps(payload, indent=2)}")

            self.generate_yaml_file(payload, output_dir)

if __name__ == "__main__":   
    processor = QnaGeneratorProcessor(model_id=os.getenv("MODEL_ID"))
    try:
        processor.process(f"{Path(__file__).resolve().parents[1]}/docs", 
                          f"{Path(__file__).resolve().parents[1]}/markdown", 
                          table_dir=f"{Path(__file__).resolve().parents[1]}//tables")
    # for input_dir, output_dir, review_dir in zip(input_dirs, output_dirs, review_dirs):
    #     processor.process(input_dir, output_dir, review_dir)
    except Exception as e:
        print("An exception occurred:")
        traceback.print_exc()
        print(f"Exception message: {e}")