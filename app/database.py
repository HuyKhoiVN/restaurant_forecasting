import pyodbc
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import settings

# Thiết lập connection string với Windows Authentication
SQLSERVER_CONNECTION_STRING = 'DRIVER={ODBC Driver 17 for SQL Server};'\
        f'SERVER=HuyKhoiTUF\\SQLEXPRESS;'\
        'DATABASE=RestaurantDB;'\
        'Trusted_Connection=yes;'

# Tạo engine SQLAlchemy
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={SQLSERVER_CONNECTION_STRING}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def connect_to_db():
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        f'SERVER=HuyKhoiTUF\\SQLEXPRESS;'
        'DATABASE=RestaurantDB;'
        'Trusted_Connection=yes;'
    )
    return conn
