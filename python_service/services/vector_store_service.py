from enum import Enum
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import torch
import json
from config import settings
from qdrant_client import AsyncQdrantClient, models, QdrantClient
from models.QdrantModel import QdrantStatus, QdrantCollection
import logging
from unstructured.cleaners.core import clean
from services.ViRanker_Compressor import ViRankerCompressor

logger = logging.getLogger(__name__)

class EmbeddingType(Enum):
    OPENAI = "openai"
    GOOGLE = "google"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    
def getEmbedding(embeddingType: EmbeddingType):
    match embeddingType:
        case EmbeddingType.OLLAMA:
            embeddings = OllamaEmbeddings(
            model=settings.OLLAMA_MODEL_EMBEDDING,
                base_url=settings.OLLAMA_BASE_URL,
            )
            return embeddings
        case default:
            raise ValueError(f"Invalid embedding type: {embeddingType}")

class VectorStoreService:
    def __init__(self):
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        logger.info(f"Initializing QdrantService for collection: {self.collection_name}")
        
        client : QdrantClient = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )
        
        if(settings.CREATE_COLLECTION):
            if(not client.collection_exists(settings.QDRANT_COLLECTION_NAME)):
                client.create_collection(settings.QDRANT_COLLECTION_NAME,
                                        vectors_config=models.VectorParams(
                                            size=768,
                                            distance=models.Distance.COSINE
                                        ))
        
        self.embedding :Embeddings = getEmbedding(EmbeddingType(settings.EMBEDDING_TYPE))
        
        # Initialize vector store with sync client for compatibility
        self.vector_store = QdrantVectorStore(
                                client=client,
                                collection_name=self.collection_name,
                                embedding=self.embedding
                            )
        
        
        self.async_client :AsyncQdrantClient= None
        
    async def get_async_client(self):
        """Get async client when needed"""
        if not self.async_client:
            self.async_client = AsyncQdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY
            )
        return self.async_client

    async def create_collection(self,collection_name = None,
                                vectors_size: int = 768, 
                                distance: models.Distance = models.Distance.COSINE) -> QdrantStatus:
        
        try:
            client = await self.get_async_client()
            exists = await client.collection_exists(collection_name=collection_name or settings.QDRANT_COLLECTION_NAME)
            if(not exists):
                res = await client.create_collection(
                    collection_name=collection_name or settings.QDRANT_COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=vectors_size,
                        distance=distance
                    )
                )
                
                logger.info(f"Collection {collection_name or settings.QDRANT_COLLECTION_NAME} created successfully")
                
                return QdrantStatus(
                    success=True,
                    message=f"Collection {collection_name or settings.QDRANT_COLLECTION_NAME} created successfully",
                    collection=QdrantCollection(
                        name=collection_name or settings.QDRANT_COLLECTION_NAME,
                        vectors_size=vectors_size,
                        distance=distance,
                        status="created"
                    )
                )
            collection_created = await client.get_collection(collection_name=collection_name or settings.QDRANT_COLLECTION_NAME)
            
            logger.info(f"Collection {collection_name or settings.QDRANT_COLLECTION_NAME} already exists")
            
            return QdrantStatus(
                success=False,
                message=f"Collection {collection_name or settings.QDRANT_COLLECTION_NAME} already exists",
                collection=QdrantCollection(
                    name=collection_name or settings.QDRANT_COLLECTION_NAME,
                    vectors_size=collection_created.config.params.vectors.size,
                    distance=collection_created.config.params.vectors.distance,
                    status="exists"
                )
            )
        except Exception as e:
            error_message = str(e)
            try:
                # Try to parse the error message as JSON
                if "Raw response content:" in error_message:
                    raw_content = error_message.split("Raw response content:")[1].strip()
                    error_message = raw_content.encode().decode('unicode_escape')
            except:
                pass
            raise Exception(error_message)

    async def check_collection_exists(self, collection_name: str) -> QdrantStatus:
        client = await self.get_async_client()
        if(await client.collection_exists(collection_name=collection_name)):
            return True
        return False
            
    async def delete_collection(self, collection_name: str) -> QdrantStatus:
        client = await self.get_async_client()
        if(await client.collection_exists(collection_name=collection_name)):
            await client.delete_collection(collection_name=collection_name)
            return QdrantStatus(
                success=True,
                message=f"Collection {collection_name} deleted successfully"
            )
        
        return QdrantStatus(
            success=False,
            message=f"Collection {collection_name} does not exist"
        )
    
    async def add_point(self,
                         documents: list[Document]) -> QdrantStatus:
        
        try:
            for document in documents:
                document.metadata.update({"content_original": document.page_content})
                document.page_content = clean(
                    document.page_content,
                    extra_whitespace=True,
                    dashes=True,
                    bullets=True,
                    trailing_punctuation=True,
                    lowercase=True
                )
            ids = await self.vector_store.aadd_documents(documents)
            
            return QdrantStatus(
                success=True,
                message=f"Points added successfully: {len(ids)} Points",
                ids=ids
            )
        except Exception as e:
            raise Exception(e)
        
        return QdrantStatus(
            success=True,
            message=f"Points added successfully",
            ids=ids
        )
        
    async def create_index_full_text(self, 
                                     collection_name: str,
                                     field_names: list[str]) -> QdrantStatus:
        
        if(not await self.check_collection_exists(collection_name)):
            return QdrantStatus(
                success=False,
                message=f"Collection {collection_name} does not exist"
            )
        
        try:
            client = await self.get_async_client()
            for field_name in field_names:
                await client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=models.TextIndexParams(
                        type="text",
                        tokenizer=models.TokenizerType.WORD,
                        min_token_len=2,
                        max_token_len=15,
                        lowercase=True,
                    )
                )
            return QdrantStatus(
                success=True,
                message=f"Index created successfully for field: {json.dumps(field_names)}"
            )
        except Exception as e:
            return QdrantStatus(
                success=False,
                message=f"Error creating index: {e}"
            )

    async def search(self, collection_name: str,
                     text_query: str, limit :int = 20,
                     score_threshold: float = 0.7,
                     filters: list[str] = None
                     ) -> list[Document]:
        """_summary_

        Args:
            collection_name (str): _description_
            text_query (str): _description_
            limit (int, optional): _description_. Defaults to 20.
            score_threshold (float, optional): _description_. Defaults to 0.7.
            filters (list[str], optional): filters is list key in metadata is indexed. Defaults to None.

        Raises:
            Exception: _description_
            Exception: _description_

        Returns:
            list[Document]: _description_
        """
        text_query = clean(
            text_query,
            extra_whitespace=True,
            dashes=True,
            bullets=True,
            trailing_punctuation=True,
            lowercase=True
        )   
        filter_must: list[models.FieldCondition]= []
        if(filters is not None and score_threshold == -1):
            for filter in filters:
                filter_must.append(
                    models.FieldCondition(
                        key=filter,
                        match=models.MatchText(text=text_query)
                    )
                )
        
        exists = await self.check_collection_exists(collection_name)
        if not exists:
            raise Exception(f"Collection {collection_name} does not exist")
            

        try:
            res = await self.vector_store.asimilarity_search_with_relevance_scores(
                query=text_query,
                k=limit,
                score_threshold= None if score_threshold == -1 else score_threshold,
                filter=models.Filter(
                    must=filter_must
                )
            )
            documents :list[Document] = []

            for document, score in res:
                document.page_content = document.metadata["content_original"]
                document.metadata.update({"score": score})
                document.metadata.pop("content_original")
                documents.append(document)
                
            return documents
        except Exception as e:
            raise Exception(e)

    async def embed_documents(self, documents: list[Document]) -> list[list[float]]:
        try:
            texts = [doc.page_content for doc in documents]
            vectors = await self.embedding.aembed_documents(texts=texts)
            return vectors
        except Exception as e:
            raise Exception(e)
    
    async def embed_query(self, query: str) -> list[float]:
        try:
            vectors = await self.embedding.aembed_query(query=query)
            return vectors
        except Exception as e:
            raise Exception(e)
    
    async def compress_documents(self, query: str,
                                 limit_trieved: int = 20,
                                 limit_compressed: int = 3,
                                 score_threshold: float = 0.7) -> list[Document]:
        """
        Compress documents using the retriever

        Args:
            query (str): The query to compress the documents
            query top 20 similar documents
            compress documents with model of huggingface cross encoder
            return top 3 documents
        Raises:
            Exception: _description_

        Returns:
            list[Document]: _description_
        """
        try:
            self.viRankerCompressor = ViRankerCompressor(top_k=limit_compressed)
        
            self.compressor = CrossEncoderReranker(
                model=HuggingFaceCrossEncoder(
                    model_name=settings.HUGGINGFACE_MODEL_RERANK,
                    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
                ),
                top_n=3
            )
            self.retriever = ContextualCompressionRetriever(
                base_compressor=self.viRankerCompressor,
                base_retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": limit_trieved, "score_threshold": score_threshold}
                )
            )
            res = await self.retriever.ainvoke(query)
            return res
        except Exception as e:
            raise Exception(e)
        

    async def search_full_text(self, query: str, limit: int = 3) -> list[Document]:
        query_clean = clean(
            query,
            extra_whitespace=True,
            dashes=True,
            bullets=True,
            trailing_punctuation=True,
            lowercase=True
        )   
        filter = models.FieldCondition(
            key="title_of_chunk",
            match=models.MatchText(text=query)
        )
        
        try:
            res = await self.vector_store.asimilarity_search(
                query=query_clean,
                k=limit,
                filter=models.Filter(
                    must=[
                        filter
                    ]
                )
            )
            
            return res
        except Exception as e:
            raise Exception(e)
        