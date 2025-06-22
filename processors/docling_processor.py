import glob
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.ocr_mac_model import OcrMacOptions
import os
import shutil

class DoclingProcessor:
    def __init__(self, **kwargs):
        """
        Initialize the DocumentConverter.
        """
        self.doc_converter = self._initialize_converter(**kwargs)

    def _initialize_converter(self):
        """Set up the appropriate DocumentConverter based on the mode."""
        return DocumentConverter()

    def process(self, input_dir: str, output_dir: str, review_dirs: list) -> None:
        """
        Convert all PDFs in the input directory to Markdown and save them to the output directory.

        Args:
            input_dir (str): Directory containing PDF files
            output_dir (str): Directory for converted Markdown files
            review_dirs (list): Directories containing outputted files that should be reviewed - currently includes "tables"
        """
        input_pdfs = glob.glob(f"{input_dir}/*.pdf")

        if not input_pdfs:
            print(f"No PDFs found in {input_dir}.")
            return
        
        os.makedirs(output_dir, exist_ok=True)

        for i in input_pdfs:
            conv_result = self.doc_converter.convert(i)
            path = f"{output_dir}/{conv_result.input.file.stem}.md"
            
            with open(path, "w") as f:
                print(f"Writing file: {f.name}...")
                f.write(conv_result.document.export_to_markdown())
                
                review_tables_dir = [dir for dir in review_dirs if dir.endswith("tables")]
                if len(review_tables_dir) and len(conv_result.document.tables):
                    print(f"Copying file to tables directory: {os.path.basename(path)}...")
                    shutil.copy(f.name, dir)
                            
if __name__ == "__main__":              
    processor = DoclingProcessor({})
    input_dirs, output_dirs, review_dirs = ["docs"], ["markdown"], ["tables"]
    for input_dir, output_dir, review_dir in zip(input_dirs, output_dirs, review_dirs):
        processor.process(input_dir, output_dir, review_dir)
