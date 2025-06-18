from typing import IO, Any, List
from pydantic import BaseModel

class PdfModel(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    
    file: Any # type IO[bytes]
    file_name: str
    chunk_size: int = 1200
    chunk_overlap: int = 250
    enable_ocr: bool = True
    extract_tables: bool = True
    languages: List[str] = ["vie", "eng"]