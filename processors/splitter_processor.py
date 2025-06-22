from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
import os

class SplitterProcessor:
    def __init__(self):
        """
        Initialize the SplitterProcessor.
        """
        self.embedding_model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = HuggingFaceTokenizer(tokenizer=AutoTokenizer.from_pretrained(self.embedding_model_id),)
        self.splitter = HybridChunker(tokenizer=self.tokenizer)
        self.min_tokens = 150

    def process(self, input_file: str) -> list:
        """
        Splits docs using semantic chunking.
        
        Args:
            input_file (str): Converted Markdown files
        Return:
            chunks (list): List of chunks generated from Markdown file
        """
        print(f"Splitting file: {input_file}...")
        if not os.path.exists(input_file):
            print(f"File {input_file} does not exist.")
            return
            
        doc = DocumentConverter().convert(source=input_file).document
        chunk_iter = self.splitter.chunk(dl_doc=doc)
        enriched_chunks = []
        for i, chunk in enumerate(chunk_iter):
            enriched_chunk = self.splitter.contextualize(chunk=chunk)
            if self.tokenizer.count_tokens(text=enriched_chunk) >= int(self.min_tokens):
                enriched_chunks.append(enriched_chunk)
        return enriched_chunks
