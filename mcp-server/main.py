import json
from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()
os.environ["DANGEROUSLY_OMIT_AUTH"] = "true"
# Initialize FastMCP server
mcp = FastMCP("document_school", stateless_http = True)

# Constants
URL_RAG_API = os.getenv("URL_RAG_API") or "http://localhost:8001"
COLLECTION_NAME = os.getenv("COLLECTION_NAME") or "my_collection"
# USER_AGENT = "weather-app/1.0"

print("URL_RAG_API: ", URL_RAG_API)

client = httpx.AsyncClient(
    base_url=URL_RAG_API,
)

# Define the document_school tool
# filter is develop in the future
@mcp.tool()
async def document_school(query: str) -> str:
    """Search for school documents, if query include subject as math, IT,... and with name school is university Thủy Lợi.
    Args:
        query: summary main content of input user (e.g. "Đề thi môn toán đại học thủy lợi")
    """
    
    try:
        # /search-text is the endpoint of the rag. It return the list[Document]
        request = {"text_query": query, 
                "limit": 20,
                "limit_compressed": 3,
                "score_threshold": 0.7}
        
        request_search_text = {
            "collection_name": COLLECTION_NAME,
            "text_query": query,
            "limit": 5,
            "score_threshold": -1, # -1 là search trên filter,khác -1 là sẽ search similar vector,
            "filters": [
                "title_of_chunk"
            ]
        }
        
        result_search_rag, result_search_title = await asyncio.gather(
            client.post("/search-rag", data=request),
            client.post("/search-text", data=request_search_text)
        )
        
        documents_title = result_search_title.json()
        content_title = []
        if len(documents_title) > 0:
            for document in documents_title:
                content_title.append(document['metadata']['content_original'] + "\n")
        content_title = "\n---\n".join(content_title)
        documents_rag = result_search_rag.json()
        if len(documents_rag) > 0:
            # Format each document's content with its metadata
            formatted_results = []
            formatted_table_markdown = []
            for doc in documents_rag:
                content = f"Content: {doc['metadata']['content_original']}\nScore: {doc['metadata']['score']}"
                formatted_results.append(content)
                formatted_table_markdown.append(doc['metadata']['table_markdown'])
            page_contents = "\n---\n".join(formatted_results)
            table_markdown = "\n---\n".join(formatted_table_markdown)
            content_rag = f"""
                this is the most relevant documents from database for this question
                context: 
                {page_contents}
                
                content of table markdown, it can be empty: 
                {table_markdown}
                
                overall content of context (it is title of chunks):
                {content_title} 
            """
            
            return "Context from RAG:\n" + content_rag
        else:
            return f"Error: API returned status code {result_search_rag.status_code}"
            
    except Exception as e:
        return f"Error: from rag api {str(e)}"
    
if __name__ == "__main__":
    mcp.run(transport="stdio")