import mysql.connector
import json
import numpy as np, pandas as pd
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
        objects TEXT
    );
    """)

    print("Table 'squares' created successfully.")

    cursor.close()
    conn.close()

"""OBJECTS
"""

class DB_API:

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
            # print("Database connection opened.")

    def close_conn(self):

        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            # print("Database connection closed.")

    def create_hist_table(self, inputs, username: str = "dchiappo", db: str = "sim_db"):

        self.open_conn()
        # Delete existing table
        self.cursor.execute("DROP TABLE IF EXISTS db_hist")
        # Create based on inputs
        self.cursor.execute(rf"""
        CREATE TABLE IF NOT EXISTS db_hist (
            id INT AUTO_INCREMENT PRIMARY KEY,
            subject_id INT,
            square_id INT,
            {", ".join([f"input_{i} VARCHAR(255) NOT NULL" for i in range(inputs)])},
            output VARCHAR(255)
        );
        """)

        print("Table 'squares' created successfully.")
        self.close_conn()

    def insert_square(self, x, y, terrain, objects=None):

        self.open_conn()

        # Storing objects as json
        obj_data = json.dumps(objects) if objects else "[]"

        try:
            self.cursor.execute("""
                    INSERT INTO squares (x, y, terrain, objects)
                    VALUES ({x}, {y}, {terrain}, {obj_data});
                    """)
            self.conn.commit()
            print(f"Square at ({x}, {y}) added into db.")
        except mysql.connector.Error as err:
            print(f"Error inserting square: {err}")
        finally:
            self.cursor.close()
            self.open_conn()  # Reopen cursor for next operation

    def insert_hist(self, subject_id, square_id, feature_count, feature_data, target_data):

        self.open_conn()  # Ensure connection is open

        # Prepare SQL statement
        sql = f"""
                INSERT INTO db_hist (subject_id, square_id, {", ".join([f"input_{i}" for i in range(feature_count)])}, output)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """

        records = [
            tuple(map(int, [subject_id, square_id] + list(feature_row) + [target]))
            for feature_row, target in zip(feature_data, target_data)
        ]

        # Execute batch insert
        self.cursor.executemany(sql, records)

        self.conn.commit()  # Save changes
        self.close_conn()  # Close connection

    def get_squares(self):

        # Fetches all squares from the database.
        self.open_conn()
        self.cursor.execute("SELECT * FROM squares")

        return self.cursor.fetchall()

    def get_hist(self, subject_id):

        self.open_conn()
        # Query and retreive
        query = "SELECT * FROM db_hist WHERE subject_id = %s;"
        self.cursor.execute(query, (subject_id,))
        hist = self.cursor.fetchall()
        columns = [col[0] for col in self.cursor.description] # Get column names
        self.close_conn()
        # Convert to pandas df
        df = pd.DataFrame(hist, columns=columns)

        return df

    def truncate_squares(self):

        # Removes all rows and resets ID counter.
        self.open_conn()
        self.cursor.execute("TRUNCATE TABLE squares")
        self.conn.commit()
        print("All squares deleted, and ID reset.")


"""RUN
"""
if __name__ == "__main__":
    db = DB_API()
    db.create_hist_table(inputs=3)