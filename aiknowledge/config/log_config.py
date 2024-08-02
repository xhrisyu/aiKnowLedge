import logging
import os


def setup_logging():
    logger = logging.getLogger('aiknowledge')
    logger.setLevel(logging.DEBUG)

    # 获取项目根目录路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    # 设置日志目录为项目根目录下的 logs 文件夹
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'app.log')

    # Clear any existing handlers to avoid duplicate logs
    # if logger.hasHandlers():
    #     logger.handlers.clear()
    # for handler in logger.handlers[:]:
    #     logger.removeHandler(handler)

    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 创建文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 创建控制台处理器
        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(logging.DEBUG)
        # console_handler.setFormatter(formatter)
        # logger.addHandler(console_handler)

    return logger
