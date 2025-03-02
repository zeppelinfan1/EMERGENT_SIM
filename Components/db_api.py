import mysql.connector
from Components.get_auth import get_auth


"""STAND ALONE FUNCTIONS
   Used for administrative tasks that aren't used during runtime i.e. altering database structure.
"""
# Database creation
def db_create(username: str="dchiappo"):

    pwd = get_auth(service_name="mysql", username=username)

    conn = mysql.connector.connect(
        host="localhost",
        user=username,
        password=pwd
    )

    cursor = conn.cursor()
    print("Connected")

    cursor.execute("CREATE DATABASE IF NOT EXISTS sim_db")

    cursor.close()
    conn.close()


def squares_table_create(username: str="dchiappo", db: str="sim_db"):

    pwd = get_auth(service_name="mysql", username=username)

    conn = mysql.connector.connect(
        host="localhost",
        user=username,
        password=pwd,
        database=db
    )

    cursor = conn.cursor()
    print("Connected")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS squares (
        id INT AUTO_INCREMENT PRIMARY KEY,
        x INT NOT NULL,
        y INT NOT NULL,
        terrain VARCHAR(50) NOT NULL,
        objects TEXT  -- Stores objects in JSON format (optional)
    );
    """)

    print("Table 'squares' created successfully.")

    cursor.close()
    conn.close()


"""OBJECTS
"""

class db_api:

    def __init__(self, username: str="dchiappo", db: str="sim_db"):

        self.username = username
        self.db = db
        self.password = get_auth(service_name="mysql", username=username)
        self.conn = None
        self.cursor = None

    def open_conn(self):

        if not self.conn or not self.conn.is_connected():
            self.conn = mysql.connector.connect(
                host="localhost",
                user=self.username,
                password=self.password,
                database=self.db
            )
            self.cursor = self.conn.cursor(dictionary=True)
            print("Database connection opened.")

    def close_conn(self):

        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            print("Database connection closed.")

    def insert_square(self, x, y, terrain, objects=None):

        pass

    # Bulk update for all created squares
    def update_squares(self):

        pass

    def get_squares(self):

        # Fetches all squares from the database.
        self.open_conn()
        self.cursor.execute("SELECT * FROM squares")

        return self.cursor.fetchall()

    def truncate_squares(self):

        # Removes all rows and resets ID counter.
        self.open_conn()
        self.cursor.execute("TRUNCATE TABLE squares")
        self.conn.commit()
        print("All squares deleted, and ID reset.")


"""RUN
"""
if __name__ == "__main__":
    pass