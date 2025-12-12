import os
from pathlib import Path
from typing import List, Dict
import pandas as pd
from pypdf import PdfReader
from docx import Document


class DocumentLoader:
    def __init__(self, documents_path: str):
        self.documents_path = Path(documents_path)
    
    def load_pdf(self, file_path: Path) -> str:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def load_docx(self, file_path: Path) -> str:
        doc = Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    
    def load_excel(self, file_path: Path) -> str:
        df = pd.read_excel(file_path)
        return df.to_string()
    
    def load_csv(self, file_path: Path) -> str:
        df = pd.read_csv(file_path)
        return df.to_string()
    
    def load_txt(self, file_path: Path) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def load_all_documents(self) -> List[Dict]:
        documents = []
        supported_extensions = {
            '.pdf': self.load_pdf,
            '.docx': self.load_docx,
            '.doc': self.load_docx,
            '.xlsx': self.load_excel,
            '.xls': self.load_excel,
            '.csv': self.load_csv,
            '.txt': self.load_txt
        }
        
        for file_path in self.documents_path.iterdir():
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in supported_extensions:
                    try:
                        content = supported_extensions[ext](file_path)
                        documents.append({
                            'filename': file_path.name,
                            'content': content,
                            'extension': ext
                        })
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
        
        return documents
