from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DB_SERVER: str = "HuyKhoiTUF\\SQLEXPRESS"  # Không có dấu `;` ở cuối
    DB_NAME: str = "RestaurantDB"

    class Config:
        env_file = ".env"

settings = Settings()
