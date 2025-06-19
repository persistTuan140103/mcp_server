import json
from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("document_school", stateless_http = True)

# Constants
URL_RAG_API = os.getenv("URL_RAG_API") or "http://localhost:8000"
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
        result_search = await client.post("/search-text", 
                                        json={"text_query": query, 
                                              'collection_name': COLLECTION_NAME,
                                              "limit": 10})
        print("result_search: ", result_search)
        
        if result_search.status_code == 200:
            documents = result_search.json()
            if len(documents) > 0:
                # Format each document's content with its metadata
                formatted_results = []
                for doc in documents:
                    content = f"Content: {doc['page_content']}\nScore: {doc['metadata']['score']}"
                    formatted_results.append(content)
                
                return "Context from RAG:\n" + "\n---\n".join(formatted_results)
            else:
                return "No relevant documents found"
        else:
            return f"Error: API returned status code {result_search.status_code}"
            
    except Exception as e:
        return f"Error: from rag api {str(e)}"
    
if __name__ == "__main__":
    mcp.run(transport="stdio")