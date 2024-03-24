"""
LOGGING DATABASE CONTROLLER
"""

# IMPORTS
import mysql.connector
import pandas as pd
from Components.get_auth import get_auth


# OBJECTS
class LOG:

    def __init__(self):

        # DB credentials and connection
        user, pwd = get_auth(address="my_credentials", username="log_db")
        self.conn = mysql.connector.connect(user=user, password=pwd, host='127.0.0.1', database='log')
        # Constants
        self.log_db_cols = ["log_id", "log_level", "log_date", "log_time", "log_type", "log"]

    def insert(self, log_level: int, log_type: str, log: str) -> None:

        pass

    def query(self, text: str) -> pd.DataFrame:

        c = self.conn.cursor()
        # Execute query and grab result
        c.execute(text)
        result = c.fetchall()

        # Store result in dataframe
        df = pd.DataFrame(data=result, columns=self.log_db_cols)

        return df


# RUN
if __name__ == "__main__":
    log = LOG()

    print(log.query(text=r"select * from main_log"))
