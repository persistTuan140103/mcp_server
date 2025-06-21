from typing import Sequence, List
from fsspec import Callback
from pydantic import BaseModel, Field, ConfigDict
from langchain.retrievers.document_compressors.base import (
    BaseDocumentCompressor,
)
from langchain.schema import Document
from FlagEmbedding import FlagReranker

class ViRankerCompressor(BaseDocumentCompressor):
    """
    ViRankerCompressor is a document compressor that uses ViRanker to compress documents.
    It is a subclass of BaseDocumentCompressor.
    get top_k documents from the documents list.
    """
    reRanker: FlagReranker = Field(default=None)
    top_k: int = Field(default=3)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, model_name: str = "namdp-ptit/ViRanker", top_k: int = 3):
        super().__init__()
        self.reRanker = FlagReranker(model_name, use_fp16=True)
        self.top_k = top_k
        
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callback | None = None,
    ) -> Sequence[Document]: 
        
        pairs = [[query, doc.page_content] for doc in documents]
        
        scores = self.reRanker.compute_score(sentence_pairs=pairs, normalize=True)

        doc_scores = list(zip(documents, scores))
        
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        compressor_documents = [doc for (doc, score) in doc_scores[:self.top_k]]
        for (doc, score) in doc_scores[:self.top_k]:
            doc.metadata["score"] = score
            
        return compressor_documents