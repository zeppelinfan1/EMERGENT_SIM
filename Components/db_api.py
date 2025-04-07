import mysql.connector
import os, json
os.environ['PYSPARK_PYTHON'] = r'D:\Python\python.exe'
os.environ['PYSPARK_DRIVER_PYTHON'] = r'D:\Python\python.exe'
import numpy as np, pandas as pd
from pyspark.sql import SparkSession
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

    cursor.execute("CREATE DATABASE IF NOT EXISTS sim_db")

    cursor.close()
    conn.close()

def features_table_create(username: str="dchiappo", db: str="sim_db"):

    pwd = get_auth(service_name="mysql", username=username)

    conn = mysql.connector.connect(
        host="localhost",
        user=username,
        password=pwd,
        database=db
    )

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS features (
        id INT AUTO_INCREMENT PRIMARY KEY,
        feature_name VARCHAR(25) NOT NULL,
        feature_type VARCHAR(25) NOT NULL,
        energy_change INT NOT NULL,
        create_prob INT NOT NULL
    );
    """)

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

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS squares (
        id INT AUTO_INCREMENT PRIMARY KEY,
        x_coordinate INT NOT NULL,
        y_coordinate INT NOT NULL
    );
    """)

    cursor.close()
    conn.close()

def square_features_create(username: str="dchiappo", db: str="sim_db"):

    pwd = get_auth(service_name="mysql", username=username)

    conn = mysql.connector.connect(
        host="localhost",
        user=username,
        password=pwd,
        database=db
    )

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS square_features (
    id INT AUTO_INCREMENT PRIMARY KEY,
    square_id INT NOT NULL,
    feature_id INT NOT NULL,
    FOREIGN KEY (square_id) REFERENCES squares(id) ON DELETE CASCADE,
    FOREIGN KEY (feature_id) REFERENCES features(id) ON DELETE CASCADE
    );
    """)

    cursor.close()
    conn.close()

def subject_table_create(username: str="dchiappo", db: str="sim_db"):

    pwd = get_auth(service_name="mysql", username=username)

    conn = mysql.connector.connect(
        host="localhost",
        user=username,
        password=pwd,
        database=db
    )

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS subjects (
        id INT NOT NULL PRIMARY KEY
    );
    """)

    cursor.close()
    conn.close()

def current_positions_table_create(username: str="dchiappo", db: str="sim_db"):

    pwd = get_auth(service_name="mysql", username=username)

    conn = mysql.connector.connect(
        host="localhost",
        user=username,
        password=pwd,
        database=db
    )

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS current_positions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        iteration INT NOT NULL,
        subject_id INT NOT NULL,
        square_id INT NOT NULL,
        square_x INT NOT NULL,
        square_y INT NOT NULL,
        FOREIGN KEY (subject_id) REFERENCES subjects(id),
        FOREIGN KEY (square_id) REFERENCES squares(id)
    );
    """)

    cursor.close()
    conn.close()

def environmental_changes_table_create(username: str="dchiappo", db: str="sim_db"):

    pwd = get_auth(service_name="mysql", username=username)

    conn = mysql.connector.connect(
        host="localhost",
        user=username,
        password=pwd,
        database=db
    )

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS environmental_changes (
        id INT AUTO_INCREMENT PRIMARY KEY,
        iteration INT NOT NULL,
        subject_id INT NOT NULL,
        square_id INT NOT NULL,
        energy_change INT NOT NULL,
        FOREIGN KEY (subject_id) REFERENCES subjects(id),
        FOREIGN KEY (square_id) REFERENCES squares(id)
    );
    """)

    cursor.close()
    conn.close()

def environmental_memory_table_create(username: str="dchiappo", db: str="sim_db"):

    pwd = get_auth(service_name="mysql", username=username)

    conn = mysql.connector.connect(
        host="localhost",
        user=username,
        password=pwd,
        database=db
    )

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS environmental_memory (
        id INT AUTO_INCREMENT PRIMARY KEY,
        iteration INT NOT NULL,
        subject_id INT NOT NULL,
        square_id INT NOT NULL,
        FOREIGN KEY (subject_id) REFERENCES subjects(id),
        FOREIGN KEY (square_id) REFERENCES squares(id)
    );
    """)

    cursor.close()
    conn.close()

def feature_memory_table_create(username: str="dchiappo", db: str="sim_db"):

    pwd = get_auth(service_name="mysql", username=username)

    conn = mysql.connector.connect(
        host="localhost",
        user=username,
        password=pwd,
        database=db
    )

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feature_memory (
        id INT AUTO_INCREMENT PRIMARY KEY,
        iteration INT NOT NULL,
        subject_id INT NOT NULL,
        target_subject_id INT NOT NULL,
        square_id INT NOT NULL,
        feature_id INT NOT NULL,
        feature_embedding TEXT NOT NULL,
        feature_label DOUBLE NOT NULL,
        FOREIGN KEY (subject_id) REFERENCES subjects(id),
        FOREIGN KEY (square_id) REFERENCES squares(id),
        FOREIGN KEY (feature_id) REFERENCES features(id)
    );
    """)

    cursor.close()
    conn.close()

def square_prediction_table_create(username: str="dchiappo", db: str="sim_db"):

    pwd = get_auth(service_name="mysql", username=username)

    conn = mysql.connector.connect(
        host="localhost",
        user=username,
        password=pwd,
        database=db
    )

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS square_prediction (
        id INT AUTO_INCREMENT PRIMARY KEY,
        iteration INT NOT NULL,
        subject_id INT NOT NULL,
        square_id INT NOT NULL,
        prediction DOUBLE NOT NULL,
        FOREIGN KEY (subject_id) REFERENCES subjects(id),
        FOREIGN KEY (square_id) REFERENCES squares(id)
    );
    """)

    cursor.close()
    conn.close()

"""OBJECTS
"""

class DB_API:

    def __init__(self, username: str="dchiappo", db: str="sim_db", spark=False, reset=True):

        self.username = username
        self.db = db
        self.password = get_auth(service_name="mysql", username=username)
        self.conn = None
        self.cursor = None

        if reset: self.reset_tables()

        # Table creation
        db_create()
        features_table_create()
        subject_table_create()
        squares_table_create()
        environmental_changes_table_create()
        current_positions_table_create()
        square_features_create()
        environmental_memory_table_create()
        feature_memory_table_create()
        square_prediction_table_create()

        # Initialize SparkSession for JDBC
        if spark:
            jar_path = r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\MySQL\mysql-connector-j-9.2.0.jar"
            self.spark = SparkSession.builder \
                .appName("EMERGENT_SIM_DB") \
                .config("spark.jars", jar_path) \
                .getOrCreate()

            self.spark.sparkContext.setLogLevel("ERROR")

            self.jdbc_url = f"jdbc:mysql://localhost:3306/{self.db}"
            self.jdbc_options = {
                "url": self.jdbc_url,
                "driver": "com.mysql.cj.jdbc.Driver",
                "user": self.username,
                "password": self.password,
            }

    def reset_tables(self):

        conn = mysql.connector.connect(
            host="localhost",
            user=self.username,
            password=self.password,
            database=self.db
        )
        cursor = conn.cursor()
        tables = ["square_prediction", "feature_memory", "environmental_memory", "current_positions",
                  "environmental_changes", "square_features", "squares", "subjects", "features"]
        for table in tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table};")
        conn.commit()
        cursor.close()
        conn.close()

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

    def insert_mysql_bulk(self, table_name: str, data: list[dict]):

        if not data:
            print(f"[INFO] No data to insert into '{table_name}'. Skipping.")
            return

        # Extract column names from the first dictionary
        columns = data[0].keys()
        placeholders = ", ".join(["%s"] * len(columns))
        column_str = ", ".join(columns)
        insert_query = f"INSERT INTO {table_name} ({column_str}) VALUES ({placeholders})"

        values = [tuple(row[col] for col in columns) for row in data]

        try:
            self.open_conn()
            self.cursor.executemany(insert_query, values)
            self.conn.commit()
        except Exception as e:
            print(f"[ERROR] MySQL insert failed for '{table_name}': {e}")
        finally:
            self.close_conn()

    def insert_dataframe_spark(self, df, table_name: str, mode: str = "append"):

        if not hasattr(self, 'spark') or self.spark is None:
            raise ValueError("SparkSession is not initialized.")

        if df.rdd.isEmpty():
            print(f"[INFO] DataFrame for '{table_name}' is empty. Skipping insert.")
            return

        try:
            df.write \
                .format("jdbc") \
                .options(**self.jdbc_options, dbtable=table_name) \
                .mode(mode) \
                .save()
            print(f"[SUCCESS] Inserted DataFrame into '{table_name}' with mode='{mode}'.")

        except Exception as e:
            print(f"[ERROR] Failed to insert into '{table_name}': {e}")

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