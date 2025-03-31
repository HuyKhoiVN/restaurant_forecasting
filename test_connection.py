from app.database import connect_to_db

def test_connection():
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("SELECT TOP 5 * FROM Revenue")
        print("Kết nối thành công! 5 bản ghi đầu từ bảng Revenue:")
        for row in cursor.fetchall():
            print(row)
        conn.close()
    except Exception as e:
        print(f"Lỗi kết nối database: {e}")
        print("Kiểm tra lại:")
        print("- SQL Server (SQLEXPRESS) đang chạy")
        print("- Tên server 'HuyKhoiTUF\\SQLEXPRESS' chính xác")

if __name__ == "__main__":
    test_connection()