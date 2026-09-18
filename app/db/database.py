import os

import psycopg
from dotenv import load_dotenv


load_dotenv()


class Database:

    def __init__(self):

        database_url = os.getenv("DATABASE_URL")

        if not database_url:
            raise ValueError(
                "DATABASE_URL is not set"
            )

        self.database_url = database_url

    def connect(self):

        return psycopg.connect(
            self.database_url,
            sslmode="require"
        )