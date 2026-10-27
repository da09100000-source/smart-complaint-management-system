import sqlite3

DB_PATH = "database/complaints.db"


def create_database():
    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_id TEXT UNIQUE,
            complaint TEXT,
            organization TEXT,
            category TEXT,
            priority TEXT,
            department TEXT,
            status TEXT
        )
    """)

    connection.commit()
    connection.close()


def save_complaint(
    complaint_id,
    complaint,
    organization,
    category,
    priority,
    department,
    status
):
    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO complaints
        (
            complaint_id,
            complaint,
            organization,
            category,
            priority,
            department,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        complaint_id,
        complaint,
        organization,
        category,
        priority,
        department,
        status
    ))

    connection.commit()
    connection.close()


create_database()
def get_complaints():
    connection = sqlite3.connect(DB_PATH)

    data = connection.execute("""
        SELECT *
        FROM complaints
        ORDER BY id DESC
    """).fetchall()

    connection.close()

    return data


def update_status(complaint_id, new_status):
    connection = sqlite3.connect(DB_PATH)

    connection.execute("""
        UPDATE complaints
        SET status = ?
        WHERE complaint_id = ?
    """, (new_status, complaint_id))

    connection.commit()
    connection.close()