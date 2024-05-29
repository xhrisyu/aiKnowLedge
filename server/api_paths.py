from config import app_config


class APIPaths:
    HOST = f"http://{app_config.get('fastapi')['host']}"
    PORT = app_config.get('fastapi')['port']

    @classmethod
    def get_full_path(cls, path):
        return f"{cls.HOST}:{cls.PORT}{path}"

    @classmethod
    def get_relative_path(cls, path):
        return path

    KB_GET = "/kb/get"
    KB_INSERT = "/kb/insert"
    KB_UPDATE = "/kb/update"
    KB_REMOVE = "/kb/remove"

    VEC_INSERT = "/vec/insert"
    VEC_GET = "/vec/get"
    VEC_REMOVE = "/vec/remove"
    VEC_SEARCH = "/vec/search"
