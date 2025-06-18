from typing import IO, Any, Union
from pydantic import BaseModel
from io import BytesIO

class DocxModel(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    
    file: Any # type IO[bytes]
    file_name: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    preserve_formatting: bool = True
