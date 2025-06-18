from typing import Optional
from pydantic import BaseModel

class QdrantCollection(BaseModel):
    name: str
    vectors_size: int
    distance: str
    status: str

class QdrantStatus(BaseModel):
    success: bool
    message: str
    collection: Optional[QdrantCollection] = None
    # error: Optional[str] = None
      
class QrantPoint(BaseModel):
    id: str
    payload: dict
    vectors: list[float]


