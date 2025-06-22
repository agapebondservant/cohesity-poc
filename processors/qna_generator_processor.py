import splitter_processor
from templates.prompts import MAIN_PROMPT
from vllm import LLM, SamplingParams
import random
    
class QnaGeneratorProcessor:
    def __init__(self, model_id: str):
        """
        Initialize the QnaGeneratorProcessor.
        """
        self.llm = LLM(model=model_id)
        
        self.num_contexts = 8 # Currently supports 8 contexts per qna.yaml file
        
        self.num_contexts_per_file = [2,2,1,1,1,1]
        
        self.splitter = splitter_processor.SplitterProcessor()
        
    def select_chunks(self, input_dir: str, table_dir: str) -> list:
        """
        Selects a subset of files from input_dir and generates contexts for each file as a list of strings.
        """
        input_files = glob.glob(f"{input_dir}/*.md")
        
        sample_size = len(self.num_contexts_per_file)
        
        context_counts = [i % len(input_files) for i in range(sample_size)]
        
        context_counts = random.shuffle(context_counts)
        
        print(f"Context counts: {context_counts}")
        
        chunks = []
        
        for i, count in enumerate(context_counts):
            chunks.append(self.splitter.split_docs(input_files[i])[:count])
            
        print(chunks)
        
        return chunks

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
            print(f"No Markdown files found in {input_dir}.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
            
        with open(f"{output_dir}/qna.yaml", "w") as f:
            print(f"Generating file: {f.name}...")
            
            self.select_chunks(input_dir, table_dir)

if __name__ == "__main__":   
    processor = QnaGeneratorProcessor(model_id="ibm-granite/granite-3.2-8b-instruct")
    processor.process("markdown", "taxonomy/knowledge/support")