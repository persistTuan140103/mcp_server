import json
from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("document_school")

# Constants
URL_RAG_API = os.getenv("URL_RAG_API")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
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
        result_seatch = await client.post("/search-text", json={"text": query, 'collection_name': COLLECTION_NAME})
        
        if(result_seatch.status_code == 200):
            res = result_seatch.json()
            page_content = res['page_content']
            if(len(page_content) > 0):
                return page_content
            else:
                return "Error: No content from search"
        else:
            return "Error: " + result_seatch.text
    except Exception as e:
        return f"Error: from rag api {e}"
    
if __name__ == "__main__":
    mcp.run(transport="stdio")