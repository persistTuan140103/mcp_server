from pydantic import BaseModel, Field
class SearchTextModel(BaseModel):
    """
    Model for search text input.
    """
    collection_name: str = Field(
        ...,
        description="The name of the collection in qdrant.",
        example="example_collection"
    )
    text_query: str = Field(
        ...,
        description="The text query to search in the collection.",
        example="example search text"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="The maximum number of results to return.",
        example=10
    )
    score_threshold: float = Field(
        default=0.5,
        ge=-1,
        le=1.0,
        description="The minimum score threshold for results. if is -1 then will use filters.",
        example=0.5
    )
    filters: dict = Field(
        default=None,
        description="Optional filter to apply to the search results.",
        example={"metadata.key": "value"}
    )
    

    class Config:
        schema_extra = {
            "example": {
                "search_text": "example search text"
            }
        }

        
class SearchRAGModel(BaseModel):
    """
    Model for search RAG input.
    """
    text_query: str = Field(
        ...,
        description="The text query to search in the collection.",
        example="example search text"
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="The maximum number of results from similar vector.",
        example=10
    )
    limit_compressed: int = Field(
        default=3,
        ge=1,
        le=100,
        description="The maximum number of compressed results to return. It is finnale result is returned.",
        example=10
    )
    score_threshold: float = Field(
        default=0.5,
        ge=-1,
        le=1.0,
        description="The minimum score threshold for results. if is -1 then will use filters.",
        example=0.5
    )