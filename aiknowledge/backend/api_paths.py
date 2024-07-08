from aiknowledge.config import app_config


class APIPaths:
    HOST = f"http://{app_config.get('fastapi')['host']}"
    PORT = app_config.get('fastapi')['port']

    @classmethod
    def get_full_path(cls, path):
        return f"{cls.HOST}:{cls.PORT}{path}"

    @classmethod
    def get_relative_path(cls, path):
        return path

    KB_GET = "/rag/get"
    KB_INSERT = "/rag/insert"
    KB_UPDATE = "/rag/update"
    KB_REMOVE = "/rag/remove"

    VEC_INSERT = "/vec/insert"
    VEC_GET = "/vec/get"
    VEC_REMOVE = "/vec/remove"
    VEC_SEARCH = "/vec/search"

    QUIZ_GENERATE = "/quiz/generate"
