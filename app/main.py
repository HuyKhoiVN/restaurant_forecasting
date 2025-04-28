from fastapi import FastAPI
from app.api import revenue, product_forecast, inventory
import logging

# Cấu hình logging
logging.basicConfig(
    level=logging.DEBUG,  # Mức log: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',  # Định dạng log
    handlers=[
        logging.StreamHandler()  # In ra terminal
    ]
)

# Tạo logger cho file main
logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(revenue.router, prefix="/api/revenue", tags=["Revenue"])
app.include_router(product_forecast.router, prefix="/api/product", tags=["product_forecast"])
app.include_router(inventory.router, prefix="/api/inventory", tags=["Inventory"])