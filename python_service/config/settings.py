from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # Redis settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # API settings
    EXTERNAL_API_URL: str = "http://localhost:8000"
    
    # Stream settings
    GROUP_NAME: str
    GROUP_NAME_RETRY: str
    STREAM_NAME: str 
    RETRY_STREAM_NAME: str
    
    UNSTRUCTURED_API_URL: str = "http://192.168.1.2:8000"
    
    # OLLAMA settings
    OLLAMA_BASE_URL: str = "http://192.168.1.2:11434"
    OLLAMA_MODEL_EMBEDDING: str = "paraphrase-multilingual:latest"
    


    #qdrant
    QDRANT_COLLECTION_NAME: str
    QDRANT_URL: str
    QDRANT_API_KEY: Optional[str] = None
    CREATE_COLLECTION: bool = False
    
    EMBEDDING_TYPE: str = "ollama"
    HUGGINGFACE_MODEL_RERANK: str = "namdp-ptit/ViRanker"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        env_file_encoding="utf-8"
    )

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings() 