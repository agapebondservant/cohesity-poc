from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

    
class SplitterProcessor:
    def __init__(self):
        """
        Initialize the SplitterProcessor.
        """
        self.text_splitter = SemanticChunker(OpenAIEmbeddings())

    def split_docs(self, input_file: str) -> list:
        """
        Splits docs using semantic chunking.

        Args:
            input_file (str): Converted Markdown files
        Return:
            chunks (list): List of chunks generated from Markdown file
        """
        if not os.path.exists(input_file):
            print(f"File {input_file} does not exist.")
            return
        with open(input_file, "r") as f:
            print(f"Splitting file: {f.name}...")
            doc = f.read()
            chunks = self.text_splitter.create_documents([doc])
            print([chunk.page_content for chunk in chunks])
            return [chunk.page_content for chunk in chunks]
