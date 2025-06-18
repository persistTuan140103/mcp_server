import asyncio
import os
from pathlib import Path
import uuid
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Optional, Dict, AsyncGenerator
import logging
from bs4 import BeautifulSoup
from fastapi import HTTPException
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from unstructured.staging.base import elements_from_base64_gzipped_json
from unstructured_client import RetryConfig, UnstructuredClient
from unstructured_client.utils.retries import BackoffStrategy
from models.DocxModel import DocxModel
from models.PdfModel import PdfModel

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
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
                
                # === METADATA VÀ DEBUG ===
                unique_element_ids=True,              # Unique IDs cho debug
                include_orig_elements=True,           # Bao gồm original elements
                
                # === OUTPUT ===
                output_format="application/json",
                encoding="utf-8"
            )
            
            # Load documents
            index = 0
            metadata_main :List[Dict] = []
            
            # Thêm metadata đặc biệt cho PDF
            async for doc in loader.alazy_load():
                if("text_as_html" in doc.metadata):
                    text = self.parse_html_table_to_text(doc.metadata["text_as_html"])
                    doc.page_content += text
                decoded_elements_serializable = self.__decode_orig_elements(doc.metadata["orig_elements"])
                    
                # res = self.__create_main_title(metadata_main, decoded_elements_serializable)                           
                # main_title = ". ".join([item["text"] for item in res])
                
                doc.metadata.pop("orig_elements")
                doc.id = str(uuid.uuid4())
                doc.metadata.update({
                    'filename': model.file_name,
                    'file_type': 'pdf',
                    'chunk_index': index,
                    'chunk_size': len(doc.page_content),
                    'processing_strategy': 'hi_res',
                    'ocr_enabled': model.enable_ocr,
                    'tables_extracted': model.extract_tables,
                    'languages': model.languages,
                    # "content_title_of_chunk": main_title,
                    "orig_elements_decoded": decoded_elements_serializable
                })
                yield doc
                index += 1
            
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
                strategy="fast",                      # DOCX không cần hi_res như PDF
                
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
                
                # === METADATA ===
                unique_element_ids=True,
                include_orig_elements=True,           # Original elements cho debug
                
                # === OUTPUT ===
                output_format="application/json",
                encoding="utf-8"
            )
            
            # Load documents
            
            
            # async for doc in loader.alazy_load():
            #     yield doc
                
            
            # Thêm metadata đặc biệt cho DOCX
            # breakpoint()
            metadata_main :List[Dict] = []
            index = 0
            async for doc in loader.alazy_load():
                
                main_title = ""
                if("text_as_html" in doc.metadata):
                    text = self.parse_html_table_to_text(doc.metadata["text_as_html"])
                    doc.page_content += text
                    
                
                decoded_elements_serializable = self.__decode_orig_elements(doc.metadata["orig_elements"])
                    
                res = self.__create_main_title(metadata_main, decoded_elements_serializable)                           
                
                metadata_main = res               
                    
                main_title = ". ".join([item["text"] for item in res])
                
                doc.metadata.pop("orig_elements")
                doc.id = str(uuid.uuid4())
                doc.metadata.update({
                    'filename': model.file_name,
                    'file_type': 'docx',
                    'chunk_index': index,
                    'chunk_size': len(doc.page_content),
                    'processing_strategy': 'fast',
                    'preserve_formatting': model.preserve_formatting,
                    'chunking_strategy': 'by_title',
                    "content_title_of_chunk": main_title,
                    "orig_elements_decoded": decoded_elements_serializable
                })
                yield doc
                index += 1
            logger.info(f"DOCX processed: {index} chunks từ {model.file_name}")
            # return documents
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Lỗi xử lý DOCX {model.file_name}: {error_message}")
            if('404' in error_message):
                raise HTTPException(
                    status_code=404,
                    detail=f"Url Unstructured API is not available {self.api_url}"
                )
            raise HTTPException(
                status_code=500,
                detail=f"Error processing DOCX file: {str(e)}"
            )

    async def post_process_chunks(self, 
                          documents: List[Document],
                          target_size: int = 600,
                          overlap: int = 100) -> List[Document]:
        """
        Post-process để chia nhỏ thêm các chunks lớn
        
        Args:
            documents: Documents từ Unstructured
            target_size: Kích thước mục tiêu cuối cùng
            overlap: Overlap cho post-processing
        
        Returns:
            List documents đã được post-process
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=target_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        final_documents = []
        
        for doc in documents:
            if len(doc.page_content) > target_size:
                # Split thành sub-chunks
                sub_chunks = text_splitter.split_documents([doc])
                
                for i, sub_chunk in enumerate(sub_chunks):
                    sub_chunk.metadata.update({
                        'parent_chunk_index': doc.metadata.get('chunk_index', 0),
                        'sub_chunk_index': i,
                        'is_sub_chunk': True,
                        'post_processed': True
                    })
                
                final_documents.extend(sub_chunks)
            else:
                doc.metadata['post_processed'] = False
                final_documents.append(doc)
        
        return final_documents

    def parse_html_table_to_text(self, html_table: str) -> str:
        """
        Chuyển bảng HTML (text_as_html từ Unstructured) thành dạng văn bản dễ hiểu.
        Trả về một chuỗi văn bản mô tả từng dòng theo định dạng "Cột1: Giá trị1, Cột2: Giá trị2"
        """
        soup = BeautifulSoup(html_table, 'html.parser')
        table = soup.find('table')
        if not table:
            return ""

        # Lấy header nếu có
        headers = []
        header_row = table.find('tr')
        if header_row:
            ths = header_row.find_all(['th', 'td'])
            headers = [th.get_text(strip=True) for th in ths]

        rows_text = []
        for row in table.find_all('tr')[1:]:  # Bỏ header
            cells = row.find_all(['td', 'th'])  # để phòng bảng không chuẩn
            values = [cell.get_text(strip=True) for cell in cells]

            # Bổ sung header nếu bị thiếu
            if len(headers) < len(values):
                headers += [f"Cột{i+1}" for i in range(len(headers), len(values))]

            row_data = [f"{headers[i]}: {values[i]}" for i in range(min(len(headers), len(values)))]
            rows_text.append(", ".join(row_data))

        return ".".join(rows_text)

    def __decode_orig_elements(self, orig_elements: str) -> List[Dict]:
        """
        Giải mã và chuyển đổi orig_elements thành dạng dictionary
        """
        try:
            orig_elements = elements_from_base64_gzipped_json(orig_elements)
            # Chuyển đổi mỗi element thành dictionary
            decoded_elements_serializable = []
            for element in orig_elements:
                if hasattr(element, 'to_dict'): # Kiểm tra xem element có phương thức to_dict không
                    decoded_elements_serializable.append(element.to_dict())
                else:
                    # Nếu không có to_dict, thử chuyển đổi một cách thủ công hoặc bỏ qua
                    logger.warning(f"Element type {type(element)} does not have to_dict() method. Skipping serialization.")
                    # Hoặc bạn có thể thử biểu diễn đơn giản hơn nếu cần
                    decoded_elements_serializable.append(str(element))
            
            return decoded_elements_serializable
        except Exception as e:
            logger.error(f"Failed to decode or serialize orig_elements: {e}")
            return [] # Gán rỗng để tránh lỗi

    def __create_main_title(self, source_main_title: List[dict], 
                            source_metadata: List[dict]) -> List[dict]:
        result = source_main_title.copy()
        list_main_title = ["Title", "ListItem"]
        
        def get_index(source_main_title: List[dict], target: dict):
            for i, item in enumerate(source_main_title):
                if(item["type"] == target["type"] and item["category_depth"] == target["category_depth"]):
                    return i
            return -1
        
        for metadata in source_metadata:
            # breakpoint()
                
            if(metadata["type"] in list_main_title):
                if(metadata["type"] == "Title" and metadata["metadata"]["category_depth"] == 0):
                    result.clear()
                if(len(result) == 0):
                    result.append({
                        "type": metadata["type"],
                        "category_depth": metadata["metadata"]["category_depth"],
                        "text": metadata["text"]
                    })
                else:
                    tmp = {
                        "type": metadata["type"],
                        "category_depth": metadata["metadata"]["category_depth"],
                        "text": metadata["text"]
                    }
                    index = get_index(result, tmp)
                    if(index == -1):
                        result.append(tmp)
                    else:
                        result = result[:(index)]
                        result.append(tmp)
                    
        return result
                            

# Utility functions
def analyze_processing_results(results: Dict[str, List[Document]]):
    """Phân tích kết quả xử lý batch"""
    
    print("=== PHÂN TÍCH KẾT QUẢ XỬ LÝ ===")
    total_chunks = 0
    
    for file_path, documents in results.items():
        file_name = os.path.basename(file_path)
        file_type = Path(file_path).suffix.lower()
        
        if documents:
            chunk_sizes = [len(doc.page_content) for doc in documents]
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            
            print(f"\n📄 {file_name} ({file_type.upper()})")
            print(f"   Chunks: {len(documents)}")
            print(f"   Avg size: {avg_size:.0f} chars")
            print(f"   Size range: {min(chunk_sizes)}-{max(chunk_sizes)} chars")
            
            # Hiển thị sample chunk
            if documents:
                sample = documents[0]
                strategy = sample.metadata.get('processing_strategy', 'unknown')
                print(f"   Strategy: {strategy}")
                print(f"   Sample: {sample.page_content[:100]}...")
            
            total_chunks += len(documents)
        else:
            print(f"\n {file_name} ({file_type.upper()}): Lỗi xử lý")
    
    print(f"\n Tổng cộng: {total_chunks} chunks từ {len(results)} files")
    
