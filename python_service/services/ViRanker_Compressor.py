from typing import Any, Dict, Sequence, List, Tuple
from fsspec import Callback
from pydantic import BaseModel, Field, ConfigDict
from langchain.retrievers.document_compressors.base import (
    BaseDocumentCompressor,
)
from langchain_community.cross_encoders.base import BaseCrossEncoder

from langchain.schema import Document
from FlagEmbedding import FlagReranker

class ViRankerCrossEncoder(BaseModel, BaseCrossEncoder):
    class Config:
        arbitrary_types_allowed = True

    """Vietnamese Ranker cross encoder using FlagEmbedding library.
    
    This model is based on BAAI/bge-m3 and fine-tuned for Vietnamese text ranking.
    It's recommended to use FlagEmbedding library as suggested by the model authors.
    
    Example:
        .. code-block:: python

            from your_module import ViRankerCrossEncoder

            model_name = "namdp-ptit/ViRanker"
            model_kwargs = {'use_fp16': True}
            reranker = ViRankerCrossEncoder(
                model_name=model_name,
                model_kwargs=model_kwargs
            )
    """

    model_name: str = "namdp-ptit/ViRanker"
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    normalize: bool = True
    client: FlagReranker = None  #: :meta private:

    def __init__(self, **kwargs: Any):
        """Initialize the ViRanker cross encoder using FlagEmbedding."""
        super().__init__(**kwargs)
        
        # try:
        #     from FlagEmbedding import FlagReranker
        # except ImportError as exc:
        #     raise ImportError(
        #         "Could not import FlagEmbedding python package. "
        #         "Please install it with `pip install FlagEmbedding`."
        #     ) from exc

        # Initialize FlagReranker with model kwargs
        self.client = FlagReranker(
            self.model_name, 
            use_fp16=True,
            **self.model_kwargs
        )

    def score(self, text_pairs: List[Tuple[str, str]]) -> List[float]:
        """Compute similarity scores using ViRanker model via FlagEmbedding.

        Args:
            text_pairs: The list of text pairs to score the similarity.

        Returns:
            List of scores, one for each pair.
        """
        if not text_pairs:
            return []
        
        # Convert tuples to lists as expected by FlagReranker
        pairs_list = [[pair[0], pair[1]] for pair in text_pairs]
        
        # Get scores using FlagReranker
        if len(pairs_list) == 1:
            # Single pair
            score = self.client.compute_score(pairs_list[0], normalize=self.normalize)
            return [float(score[0])]
        else:
            # Multiple pairs
            scores = self.client.compute_score(pairs_list, normalize=self.normalize)
            return [float(score) for score in scores]
