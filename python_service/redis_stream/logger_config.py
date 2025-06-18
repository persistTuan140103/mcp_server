import logging
import sys

def setup_logging():
    """
    Cấu hình Python's logging module để tách biệt log INFO/WARNING/DEBUG
    và log ERROR/CRITICAL sang các luồng stdout và stderr riêng biệt.
    """
    # Lấy logger gốc (hoặc logger cụ thể cho ứng dụng của bạn)
    # Nếu bạn muốn log từ tất cả các module trong ứng dụng, hãy dùng logging.getLogger()
    # Nếu bạn muốn log từ từng module cụ thể, hãy dùng logging.getLogger(__name__)
    logger = logging.getLogger() # Đây là root logger, sẽ ảnh hưởng đến tất cả các log
    logger.setLevel(logging.INFO) # Đặt cấp độ log tối thiểu cho logger (mọi thứ từ INFO trở lên sẽ được xử lý)

    # Xóa tất cả các handler hiện có để tránh log bị ghi trùng lặp
    # (Quan trọng để tránh các handler mặc định hoặc handler cũ)
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- Handler cho các thông báo INFO, WARNING, DEBUG (đi vào stdout) ---
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO) # Handler này chỉ xử lý từ INFO trở lên
    stdout_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stdout_handler.setFormatter(stdout_formatter)
    logger.addHandler(stdout_handler)

    # --- Handler cho các thông báo ERROR, CRITICAL (đi vào stderr) ---
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR) # Handler này chỉ xử lý từ ERROR trở lên
    stderr_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stderr_handler.setFormatter(stderr_formatter)
    logger.addHandler(stderr_handler)

    # --- Tùy chọn: Xử lý các log từ các thư viện bên ngoài (ví dụ: redis, aiohttp) ---
    # Các thư viện này thường có logger riêng của chúng. Bạn có thể điều chỉnh cấp độ của chúng.
    # logging.getLogger('redis').setLevel(logging.WARNING)
    # logging.getLogger('aiohttp').setLevel(logging.WARNING)

    return logger

# Gọi hàm setup_logging khi module này được import
# logger = setup_logging() # Uncomment dòng này nếu bạn muốn sử dụng logger đã cấu hình trực tiếp từ đây