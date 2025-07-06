import asyncio
from contextlib import asynccontextmanager
import io
import warnings
from fastapi import Body, FastAPI, Form, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse
from models.DocxModel import DocxModel
from models.PdfModel import PdfModel
from services.file_processor import DocumentProcessor
from config import settings
from langchain.schema import Document
from typing import Any, List
import logging
from services import RedisService, VectorStoreService
import json
from redis_stream import setup_logging
from models.search_model import SearchRAGModel, SearchTextModel
from fastapi.middleware.cors import CORSMiddleware


from starlette.middleware.base import BaseHTTPMiddleware

class TimeoutMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            # Thiết lập timeout cho yêu cầu
            response = await asyncio.wait_for(call_next(request), timeout=600)  # Timeout 300 giây
            return response
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={"detail": "Request timed out"}
            )



setup_logging()
logger = logging.getLogger(__name__)

def deprecated(func):
    def wrapper(*args, **kwargs):
        warnings.warn(f"{func.__name__} is deprecated", DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    return wrapper

async def init_services():
    try:
        document_processor = DocumentProcessor(settings.UNSTRUCTURED_API_URL)
        vector_store_service = VectorStoreService()
        redis_service = RedisService()
        ping_redis = await redis_service.ping()
        if(ping_redis):
            logger.info(f"Services 'Redis' initialized successfully")
        else:
            logger.error(f"Services 'Redis' initialized failed")
            raise HTTPException(status_code=500, detail=f"Services 'Redis' initialized failed")
        logger.info(f"Services 'Unstructured' initialized successfully")
        logger.info(f"Services 'VectorStore' initialized successfully")
        return document_processor, vector_store_service, redis_service
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        raise HTTPException(status_code=500, detail=f"Error initializing services: {e}")


# Khởi tạo services
document_processor: DocumentProcessor= None
vector_store_service: VectorStoreService = None
redis_service: RedisService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global document_processor, vector_store_service, redis_service
    document_processor, vector_store_service, redis_service = await init_services()
    yield
    
app = FastAPI(
    lifespan=lifespan,
    # openapi_url="/openapi.json",
    )
app.add_middleware(TimeoutMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/process-file-docx", response_model=List[Document])
async def process_file_docx(
    file: UploadFile = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    preserve_formatting: bool = Form(True),
    get_max_documents: int = Form(20)
) -> List[Document]:
    
    supported_types = [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
        "application/msword"  # .doc
    ]
    
    if file.content_type in supported_types:
        file_content = await file.read()
        file_name = file.filename.split('.')[0]  # Extract the file name without extension
        file_obj = io.BytesIO(file_content)
        
        index = 0
        async for batch_page in document_processor.split_by_page_doc(file_obj):
            docx_model = DocxModel(
                file=batch_page,
                file_name=file_name + f"_{index}.docx",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                preserve_formatting=preserve_formatting
            )
            index += 1
            try:
                # documents = await document_processor.process_docx_document(docx_model)
                lengthDocuments = 0
                tmp_documents: List[Document] = []
                async for doc in document_processor.process_docx_document(docx_model):
                    if(lengthDocuments < get_max_documents or get_max_documents == -1):
                        # print("Document: ", doc)
                        doc.metadata.pop("orig_elements_decoded") # orig support for test
                        tmp_documents.append(doc)
                    else:
                        return tmp_documents
                    lengthDocuments += 1
                return tmp_documents
            except Exception as e:
                logger.error(f"Processing Docx: {e}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Error processing DOCX file: {e}"
                )
    else:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file type. Please upload a DOC or DOCX file."
        )
    
@app.post("/process-file-pdf", response_model=List[Document])
async def process_file_pdf(
    file: UploadFile = File(...),
    chunk_size: int = Form(1200),
    chunk_overlap: int = Form(250),
    enable_ocr: bool = Form(True),
    extract_tables: bool = Form(True),
    get_max_documents: int = Form(20),
    batch_page: int = Form(10)
) -> List[Document]:
    documents = []
    if(file.content_type == "application/pdf"):
        file_content = await file.read()
        file_name = file.filename.split('.')[0]  # Extract the file name without extension
        file_obj = io.BytesIO(file_content)
        index = 0
        async for batch_page in document_processor.split_by_page(file_obj, batch_page=batch_page):
            pdf_model = PdfModel(
                file=batch_page,
                file_name=file_name + f"_{index}.pdf",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                enable_ocr=enable_ocr,
                extract_tables=extract_tables
            )
            index += 1
            lengthDocuments = 0
            try:
                async for doc in document_processor.process_pdf_document(pdf_model):
                    if(lengthDocuments < get_max_documents or get_max_documents == -1):
                        # doc.metadata.pop("orig_elements_decoded") # orig support for testing
                        documents.append({
                            "page_content": doc.page_content,
                            "metadata": doc.metadata
                        })
                    # else:
                        # return documents
                    lengthDocuments += 1
            except Exception as e:
                logger.error(f"Processing PDF: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Error processing PDF file: {e}"
                )
        return documents
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF file.")

@app.post("/process-file-pdf-redis", response_model=List[Document])
async def process_file_pdf_redis(
    chunk_size: int = Form(1200),
    chunk_overlap: int = Form(250),
    enable_ocr: bool = Form(True),
    extract_tables: bool = Form(True),
    file: UploadFile = File(...),
    get_max_documents: int = Form(20),
    batch_page_size: int = Form(10)
) -> List[Document]:
    documents = []
    lengthDocuments = 0
    if(file.content_type == "application/pdf"):
        file_content = await file.read()
        file_name = file.filename.split('.')[0]  # Extract the file name without extension
        file_obj = io.BytesIO(file_content)
        index = 0
        async for batch_page in document_processor.split_by_page(file_obj, batch_page=batch_page_size):
            pdf_model = PdfModel(
                file=batch_page,
                file_name=file_name + f"_{index}.pdf",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                enable_ocr=enable_ocr,
                extract_tables=extract_tables
            )
            index += 1
        
            try:
                
                async for doc in document_processor.process_pdf_document(pdf_model):
                    doc.metadata.pop("orig_elements_decoded", None)
                    message = {
                        "page_content": doc.page_content,
                        "metadata": json.dumps(doc.metadata, ensure_ascii=False)
                    }
                    await redis_service.add_to_stream(
                        message=message,
                        stream_name=redis_service.STREAM_NAME
                    )
                    if(lengthDocuments < get_max_documents or get_max_documents == -1):
                        documents.append(doc)
                    lengthDocuments += 1
            except asyncio.CancelledError:
                logger.info("Request cancelled by the client")
                raise HTTPException(status_code=499, detail="Client closed request")
            
            except Exception as e:
                logger.error(f"Error processing PDF file: {e}")
                raise HTTPException(status_code=500, detail=f"Error processing PDF file: {e}")
            logger.info(f"send {lengthDocuments} documents to redis")
        logger.info(f"process PDF file completed with {index} batch pages")
        serializable_documents = [doc.model_dump() for doc in documents]
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Processed {index * batch_page_size} pages from PDF file.",
                "documents": serializable_documents
            }
        )
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF file.")

@app.post("/process-file-docx-redis", response_model=List[Document])
async def process_file_docx_redis(
    chunk_size: int = Form(1200),
    chunk_overlap: int = Form(250),
    preserve_formatting: bool = Form(True),
    file: UploadFile = File(...),
    get_max_documents: int = Form(20)
) -> List[Document]:
    documents = []
    index = 0
    supported_types = [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
        "application/msword"  # .doc
    ]
    if(file.content_type in supported_types):
        file_content = await file.read()
        file_name = file.filename
        file_obj = io.BytesIO(file_content)
        docx_model = DocxModel(
            file=file_obj,
            file_name=file_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            preserve_formatting=preserve_formatting
        )
        try:
            async for doc in document_processor.process_docx_document(docx_model):
                print(f"Nhận chunk index: {index}")
                doc.metadata.pop("orig_elements_decoded")
                message = {
                    "page_content": doc.page_content,
                    "metadata": json.dumps(doc.metadata, ensure_ascii=False)
                }   
                # print(message)
                res = await redis_service.add_to_stream(
                    message=message,
                    stream_name=redis_service.STREAM_NAME
                )
                if(res is None):
                    logger.error(f"Error adding message to {redis_service.STREAM_NAME}")
                if(index < get_max_documents or get_max_documents == -1):
                    documents.append(doc)
                index += 1
        except asyncio.CancelledError:
            logger.info("Request cancelled by the client")
            raise HTTPException(status_code=499, detail="Client closed request")
        
        except Exception as e:
            logger.error(f"Error processing DOCX file: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing DOCX file: {e}")
        logger.info(f"send {index} documents to redis")
        return documents
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a DOC or DOCX file.")


@app.post("/embedd-text")
async def embedd_text(
    text: str = Body(...),
) -> Any:
    try:
        return await vector_store_service.embed_query(text)
    except Exception as e:
        logger.error(f"Error embedding text: {e}")
        raise HTTPException(status_code=500, detail=f"Error embedding text: {e}")

@deprecated
@app.post("/embed-documents")
async def embed_documents(
    documents: List[Document] = Body(...),
) -> Any:
    try:
        return await vector_store_service.embed_documents(documents)
    except Exception as e:
        logger.error(f"Error embedding documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error embedding documents: {e}")
    

@app.post("/embedd-and-save")
async def embedd_and_save(
    documents: list[Document] = Body(...)
) -> Any:
    try:
        res = await vector_store_service.add_point(documents)
        logger.info(f"log /embedd-and-save: {res}")
        return res
    except Exception as e:  
        logger.error(f"Error embedding and saving documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error embedding and saving documents: {e}")


@app.post("/create-collection")
async def create_collection(
    collection_name: str = None
) -> Any:
    try:
        return await vector_store_service.create_collection(collection_name) 
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": str(e)
            }
        )

@app.post("/search-text", description="Search for documents in a collection using a text query. If score_threshold is not provided or less than or equal to 0, it will return all documents that match the query.")
async def search(
    request: SearchTextModel,
) -> list[Document]:
    collection_name: str = request.collection_name
    text_query: str = request.text_query
    limit: int = request.limit
    score_threshold: float = request.score_threshold
    filters: dict[str, str] = request.filters
    try:
        if(score_threshold <= 0):
            if(len(filters.items()) > 0):
                res = await vector_store_service.search_with_filter(
                    collection_name=collection_name,
                    filters=filters,
                    limit=limit,
                )
                return res
            else:
                raise HTTPException(
                    status_code=400,
                    detail="filters must be provided when score_threshold is less than or equal to 0"
                )
        elif(score_threshold > 1):
            raise HTTPException(
                status_code=400, 
                detail="score_threshold must be between 0 and 1"
            )
        res = await vector_store_service.search(
            collection_name, text_query, 
            limit=limit, score_threshold=score_threshold)
        return res
    except Exception as e:
        logger.error(f"Error searching: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching: {e}")

@app.post("/search-rag")
async def search_rag(
    request: SearchRAGModel,
) -> list[Document]:
    text_query: str = request.text_query
    limit: int = request.limit
    limit_compressed: int = request.limit_compressed
    score_threshold: float = request.score_threshold
    try:
        res = await vector_store_service.compress_documents(
            text_query, limit_trieved=limit, 
            limit_compressed=limit_compressed, score_threshold=score_threshold)
        return res
    except Exception as e:
        logger.error(f"Error searching RAG: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching RAG: {e}")

@app.post("/create-index")
async def create_index(
    collection_name: str = Body(...),
    field_names: list[str] = Body(...)
) -> Any:
    try:
        return await vector_store_service.create_index_full_text(collection_name, field_names)
    except Exception as e:
        logger.error(f"Error creating index: {e}")
    
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=True,  # Enable auto-reload
    )
