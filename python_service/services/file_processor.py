import asyncio
import uuid
import html2text
from langchain_unstructured import UnstructuredLoader
from langchain.schema import Document
from typing import List, Optional, Dict, AsyncGenerator
import logging
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from unstructured.staging.base import elements_from_base64_gzipped_json
from unstructured_client import RetryConfig, UnstructuredClient
from unstructured_client.utils.retries import BackoffStrategy
from models.DocxModel import DocxModel
from models.PdfModel import PdfModel
from redis_stream import setup_logging
# from markdownify import markdownify as md


setup_logging()
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, 
                 unstructured_api_url: str = "http://localhost:8000",
                 api_key: Optional[str] = None):
        """
        Khởi tạo processor với Unstructured Docker API
        
        Args:
            unstructured_api_url: URL của Unstructured API (Docker)
            api_key: API key nếu cần (có thể None cho local)
        """
        self.api_url = unstructured_api_url
        self.api_key = api_key or ""
        
        self.client = UnstructuredClient(
            server_url=unstructured_api_url,
            api_key_auth=self.api_key,
            timeout_ms=30000, # 30s
            retry_config=RetryConfig(
                retry_connection_errors=True,
                strategy="backoff",
                backoff=BackoffStrategy(
                    initial_interval=500,
                    max_interval=60000, # 60s
                    exponent=1.5,
                    max_elapsed_time=900000 # 15m
                )
            )
        )
        
        
    @retry(
        wait=wait_fixed(10),  # Đợi 10 giây trước khi thử lại
        stop=stop_after_attempt(3), # Thử lại tối đa 3 lần
        retry=retry_if_exception_type(asyncio.TimeoutError) | retry_if_exception_type(Exception) # Có thể tùy chỉnh loại lỗi
    )
    async def process_pdf_document(self, 
                           model: PdfModel) -> AsyncGenerator[Document, None]:
        """
        Xử lý file PDF với OCR và extract tables/images
        
        Args:
            file_path: Đường dẫn tới file PDF
            chunk_size: Kích thước chunk tối đa
            chunk_overlap: Độ overlap giữa các chunk
            enable_ocr: Bật OCR cho images trong PDF
            extract_tables: Extract tables từ PDF
            languages: Danh sách ngôn ngữ cho OCR
        
        Returns:
            List các Document đã được chunk
        """
        try:
            logger.info(f"Đang xử lý PDF: {model.file_name}")
            
            # Cấu hình tối ưu cho PDF
            loader = UnstructuredLoader(
                kwargs={
                    "filename": model.file_name,
                    "split_pdf_concurrency_level": 10
                },
                file=model.file,
                timeout_ms=10000,
                client=self.client,
                partition_via_api=True,
                
                # === STRATEGY TỐI ƯU CHO PDF ===
                strategy="hi_res",                    # Chất lượng cao cho PDF
                hi_res_model_name="yolox",   # Model layout tốt nhất
                
                # === OCR VÀ NGÔN NGỮ ===
                languages=model.languages,                  # Hỗ trợ tiếng Việt + English
                coordinates=True,                     # Lấy tọa độ để debug layout
                
                # === PDF SPECIFIC ===
                # pdf_infer_table_structure=model.extract_tables,  # Infer cấu trúc bảng
                extract_image_block_types=["Image", "Table"] if model.enable_ocr else ["Table"],
                skip_infer_table_types=[],            # Không skip table types
                include_page_breaks=True,             # Giữ page breaks
                
                # === CHUNKING BY_TITLE CHO PDF ===
                chunking_strategy="by_title",         # Chunk theo cấu trúc semantic
                max_characters=model.chunk_size,            # Hard limit
                new_after_n_chars=model.chunk_size - model.chunk_overlap,  # Soft limit
                combine_under_n_chars=150,            # Kết hợp chunks nhỏ (PDF thường có nhiều đoạn ngắn)
                overlap=model.chunk_overlap,                # Overlap để đảm bảo context
                overlap_all=True,                     # Apply overlap cho tất cả
                multipage_sections=True,              # Cho phép sections span nhiều trang
                split_pdf_concurrency_level=10,         # Concurrency cho PDF processing
                split_pdf_page= True,                  # Split theo page cho PDF
                
                
                # === METADATA VÀ DEBUG ===
                unique_element_ids=True,              # Unique IDs cho debug
                include_orig_elements=True,           # Bao gồm original elements
                
                # === OUTPUT ===
                output_format="application/json",
                encoding="utf-8"
            )
            
            # Load documents
            index = 0
            tmp_document: Document = None
            # Thêm metadata đặc biệt cho PDF
            async for doc in loader.alazy_load():
                decoded_elements_serializable = self.__decode_orig_elements(doc.metadata["orig_elements"])
                    
                title_of_chunk = self.__create_main_title(decoded_elements_serializable)
                text_as_html = self.__get_text_as_html(decoded_elements_serializable)
                if(title_of_chunk == "" and tmp_document != None):
                    title_of_chunk = tmp_document.metadata["title_of_chunk"]
                
                doc.metadata.pop("orig_elements")
                doc.id = str(uuid.uuid4())
                doc.metadata.update({
                    'filename': model.file_name,
                    'file_type': 'pdf',
                    'table_markdown': self.__convert_table_markdown(text_as_html),
                    "title_of_chunk": title_of_chunk,
                    "orig_elements_decoded": decoded_elements_serializable
                })
                yield doc
                index += 1
                tmp_document = doc
            
            logger.info(f"PDF processed: {index} chunks từ {model.file_name}")
            # return documents
            
        except Exception as e:
            logger.error(f"Lỗi xử lý PDF {model.file_name}: {str(e)}")
            raise e

    async def process_docx_document(self, 
                            model: DocxModel) -> AsyncGenerator[Document, None]:
        """
        Xử lý file DOCX với tối ưu cho cấu trúc văn bản
        
        Args:
            file_path: Đường dẫn tới file DOCX
            chunk_size: Kích thước chunk tối đa
            chunk_overlap: Độ overlap giữa các chunk
            preserve_formatting: Giữ nguyên formatting
        
        Returns:
            List các Document đã được chunk
        """
        try:
            logger.info(f"Đang xử lý DOCX: {model.file_name}")
            
            # Cấu hình tối ưu cho DOCX
            loader = UnstructuredLoader(
                file=model.file,
                filename=model.file_name,
                client=self.client,
                partition_via_api=True,
                model="element",
               
                # === STRATEGY TỐI ƯU CHO DOCX ===
                strategy="auto",                      # DOCX không cần hi_res như PDF
                
                # === DOCX SPECIFIC ===
                include_page_breaks=model.preserve_formatting,  # Giữ page breaks nếu cần
                xml_keep_tags=False,                  # Không giữ XML tags
                
                # === CHUNKING BY_TITLE - TỐI ƯU CHO DOCX ===
                chunking_strategy="by_title",         # Chunk theo tiêu đề/cấu trúc
                max_characters=model.chunk_size,            # Hard limit nhỏ hơn PDF
                new_after_n_chars=model.chunk_size - model.chunk_overlap,  # Soft limit
                combine_under_n_chars=100,            # DOCX có cấu trúc tốt hơn PDF
                overlap=model.chunk_overlap,                # Overlap vừa phải
                overlap_all=True,                     # Apply overlap
                multipage_sections=True,              # Sections có thể span pages
                language=["eng", "vie"],          
                
                # === METADATA ===
                unique_element_ids=True,
                include_orig_elements=True,           # Original elements cho debug
                
                # === OUTPUT ===
                output_format="application/json",
                encoding="utf-8"
            )
            
            index = 0
            tmp_document : Document = None
            async for doc in loader.alazy_load():
                # breakpoint()
                decoded_elements_serializable = self.__decode_orig_elements(doc.metadata["orig_elements"])
                # print(decoded_elements_serializable)
                res = self.__create_main_title(decoded_elements_serializable)                           
                
                if(res == "" and tmp_document is not None):
                    res = tmp_document.metadata["title_of_chunk"]
                text_as_html = self.__get_text_as_html(decoded_elements_serializable)
                doc.metadata.pop("orig_elements")
                doc.id = str(uuid.uuid4())
                doc.metadata.update({
                    'filename': model.file_name,
                    'file_type': 'docx',
                    "title_of_chunk": res,
                    "orig_elements_decoded": decoded_elements_serializable,
                    "table_markdown": self.__convert_table_markdown(text_as_html)
                })
                print(f"Gửi chunk index: {index}")
                yield doc
                index += 1
                tmp_document = doc
            logger.info(f"DOCX processed: {index} chunks từ {model.file_name}")
            # return documents
            
        except Exception as e:
            raise e

    def __decode_orig_elements(self, orig_elements: str) -> List[Dict]:
        """
        Giải mã và chuyển đổi orig_elements thành dạng dictionary
        """
        try:
            orig_elements = elements_from_base64_gzipped_json(orig_elements)
            decoded_elements_serializable = []
            for element in orig_elements:
                if hasattr(element, 'to_dict'): # Kiểm tra xem element có phương thức to_dict không
                    decoded_elements_serializable.append(element.to_dict())
                else:
                    try:
                        # Try converting to a dictionary using vars()
                        decoded_elements_serializable.append(vars(element))
                    except TypeError:
                        # If vars() fails, try using __dict__
                        try:
                            decoded_elements_serializable.append(element.__dict__)
                        except:
                            # If all else fails, convert the object to string
                            decoded_elements_serializable.append(str(element))
            
            return decoded_elements_serializable
        except Exception as e:
            logger.error(f"Failed to decode or serialize orig_elements: {e}")
            return []
    def __create_main_title(self, 
                            elements: List[dict]) -> str:
        res = []
        for item in elements:
            if(item["type"] == "Title"):
                res.append(item["text"])
        
        return ".".join(res)
                            
    def __get_text_as_html(self, elements: list[Dict]) -> str:
        result = ""
        for item in elements:
            
            if("Table" == item["type"]):
                result += item["metadata"]["text_as_html"] + "\n"
        return result
 
    def __convert_table_markdown(self, html_content):
        """
        Convert HTML table to markdown, remove images and styles
        """
        converter = html2text.HTML2Text()
        converter.ignore_links = True  # Bỏ qua link
        converter.ignore_images = True  # Bỏ qua hình ảnh
        converter.body_width = 0  # Không giới hạn độ rộng của văn bản
        converter.unicode_snob = True  # Sử dụng ký tự Unicode
        converter.ignore_emphasis = False  # giữ **bold** và *italic*
        converter.single_line_break = True  # Chuyển đổi các dòng mới thành một dòng mới trong markdown
        markdown = converter.handle(html_content)
        
        return markdown
        
    