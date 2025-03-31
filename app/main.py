from fastapi import FastAPI
from app.api import revenue  # Thêm dòng này

app = FastAPI()

app.include_router(revenue.router, prefix="/api/revenue", tags=["Revenue"])