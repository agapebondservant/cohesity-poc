import os
import textwrap
import re
import html

class QnaCleanupProcessor:
    def __init__(self, source_dir: str = None):
        """
        Initialize the QnaCleanupProcessor.
        """
        self.source_dir = source_dir

    def process(self) -> list:
        """
        Replaces special characters with appropriate tokens.
        """
        if not source_dir:
            return None
        try:

            for root, _, source_files in os.walk(source_dir):
                
                for source_file in source_files:
                    
                    print(f"Performing cleanup on {input_file}...")
                    
                    
        except Exception as e: 
            print(f"Error loading docs: {e}") 
            
            traceback.print_exc()
            
    def process_chunk(self, chunk) -> str:
        """
        Performs cleanup at the chunk level.
        """
        lines = chunk.splitlines()
        lines = [self.cleanup_line(line) for line in lines]
        return "\n".join(lines)
    
    def cleanup_line(self, line) -> str:
        """
        - Replaces special characters with appropriate tokens.
        - Performs yaml-related cleanup.
        """
        line = line.rstrip() # strip trailing spaces
        line = html.unescape(line) # unescape HTML entities
        line = re.sub(r"([\S\n\r\\n\\r]+)(\s\s+)([\S]*?)", r"\1 \3", line) # remove multiple whitespaces between words
        return line
        
    
if __name__ == "__main__":  
    
    source_dir = f"{os.path.expanduser('~')}/{os.getenv('APP_NAME')}/scraped/studentaid"
    
    processor = QnaCleanupProcessor(source_dir)
    
    processor.process()
