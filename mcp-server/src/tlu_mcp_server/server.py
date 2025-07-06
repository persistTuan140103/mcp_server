import httpx
from mcp.server.fastmcp import FastMCP, Context
from mcp.types import TextContent
import os
from dotenv import load_dotenv
import asyncio
from asyncio import wait_for
from tlu_mcp_server.models import SearchRAGModel, SearchTextModel
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mcp_server.log'),  #
    ]
)

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.DEBUG)

note_result = """
        Dưới đây là các tài liệu được truy xuất từ hệ thống RAG, đã được sắp xếp lại bằng mô hình reranker ngữ nghĩa để ưu tiên độ liên quan cao nhất với truy vấn.
        Tuy nhiên, nội dung có thể vẫn chưa hoàn toàn chính xác hoặc đầy đủ.
        Hãy tự đánh giá độ tin cậy và phản hồi lại cho người dùng một cách rõ ràng.
        """

logger = logging.getLogger(__name__)
load_dotenv()
os.environ["DANGEROUSLY_OMIT_AUTH"] = "true"
os.environ["MCP_TIMEOUT"] = "300000"  # Set MCP timeout to 5 minutes
# Constants
URL_RAG_API = os.getenv("URL_RAG_API") or "http://127.0.0.1:8001"
COLLECTION_NAME = os.getenv("COLLECTION_NAME") or "my_collection"
# USER_AGENT = "weather-app/1.0"

logger.info(f"URL_RAG_API: {URL_RAG_API}")
# Initialize FastMCP server
mcp = FastMCP("document_school")

client = httpx.AsyncClient(
            base_url=URL_RAG_API, 
            timeout=httpx.Timeout(
                connect=60.0, 
                read=3000.0, 
                write=300.0, 
                pool=300),
            )
            

# Define the document_school tool
# filter is develop in the future
@mcp.tool(description="Search for school documents, if query include subject as math, IT,... and with name school is university Thủy Lợi.")
async def document_school(query: str) -> str:
    """Search for school documents, if query include subject as math, IT,... and with name school is university Thủy Lợi.
    Args:
        query: The main content of the input user, excluding the name of the school (e.g., "Đề thi môn toán").
               Ensure that the query does not include the phrase 'đại học Thủy Lợi' to avoid low scores from the reranker.
    """
    logger.info(f"Received query: {query}")
    try:
        # /search-text is the endpoint of the rag. It return the list[Document]
        request: SearchRAGModel= SearchRAGModel(
            text_query=query,
            limit=10,            # This is the number of results to return
            limit_compressed=3,  # The maximum number of compressed results to return. It is final result is returned.
            score_threshold=0.7,  # -1 means search on filter, different from -1 means search similar vector
        )
        
        request_search_text: SearchTextModel= SearchTextModel(
            collection_name=COLLECTION_NAME,
            text_query="",
            limit=5,            # This is the number of results to return
            score_threshold=-1,  # -1 means search on filter, different from -1 means search similar vector
            filters={"metadata.title_of_chunk": query}        
        )
        logger.info("Starting search operations...")
        # result_search_rag, result_search_title = await asyncio.gather(
        #             asyncio.wait_for(client.post("/search-rag", json=request.model_dump()), timeout=3000.0),
        #             asyncio.wait_for(client.post("/search-text", json=request_search_text.model_dump()), timeout=300.0),
        #         )
        headers = {
            "Content-Type": "application/json",
            # "User-Agent": USER_AGENT
            "encoding": "utf-8"
        }
        logger.info("Sending search requests to the RAG API...")
        result_search_rag, result_search_title = await asyncio.gather(
                client.post("/search-rag", json=request.model_dump(), headers=headers),
                client.post("/search-text", json=request_search_text.model_dump(), headers=headers)
            )
        logger.info("Search operations completed.")
        
        documents_title = result_search_title.json()
        tmp = []
        if len(documents_title) > 0:
            for document in documents_title:
                tmp.append(document['page_content']+ "\n")
        content_title = "\n---\n".join(tmp)
        documents_rag = result_search_rag.json()
        if len(documents_rag) > 0:
            # Format each document's content with its metadata
            formatted_results = []
            formatted_table_markdown = []
            for doc in documents_rag:
                page_content = doc['page_content']
                if(page_content is None or page_content == ""):
                    continue
                content = f"Content from chunk in vector DB: {page_content}"
                formatted_results.append(content)
                if( 'table_markdown' in doc['metadata'] and doc['metadata']['table_markdown'] is not None):
                    formatted_table_markdown.append(doc['metadata']['table_markdown'])
            page_contents = "\n---\n".join(formatted_results)
            table_markdown = "\n---\n".join(formatted_table_markdown)
            content_rag = f"""
                {note_result}
                context: 
                {page_contents}
                
                content of table markdown, it can be empty: 
                {table_markdown}
                
                overall content of context (it is title of chunks):
                {content_title} 
            """

            return TextContent(
                type="text",
                text=f"{content_rag}",
            ) 
        else:
            return TextContent(
                type="text",
                text="No relevant documents found in the database for this question."
            )
    except httpx.TimeoutException as e:
        logger.error(f"Timeout after {e.request.timeout} seconds")
    except TimeoutError as e:
        logger.error(f"Search operation timed out.: {type(e).__name__} - {e!r}")
        return TextContent(
            type="text",
            text="Search operation timed out. Please try again later."
        )
    
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {type(e).__name__} - {e!r}")
        return TextContent(
            type="text",
            text=f"HTTP error: {type(e).__name__} - {e!r}"
        )

    except Exception as e:
        logger.error(f"Error: {type(e).__name__} - {e!r}")
        return TextContent(
            type="text",
            text=f"An error occurred: {type(e).__name__} - {e!r}"
        )
    
def main():
    try:
        mcp.run()
    except Exception as e:
        logger.error(f"An error occurred: {type(e).__name__} - {e!r}")
        raise e

if __name__ == "__main__":
    main()