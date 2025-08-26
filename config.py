from dotenv import load_dotenv
import os

load_dotenv()

PSQL_DB_USERNAME = os.getenv("PSQL_DB_USERNAME")
PSQL_DB_PWD = os.getenv("PSQL_DB_PWD")
PSQL_DB_HOSTNAME = os.getenv("PSQL_DB_HOSTNAME")
PSQL_DB = os.getenv("PSQL_DB")
PSQL_DB_PORT = os.getenv("PSQL_DB_PORT")
PSQL_TABLE = os.getenv("PSQL_TABLE")
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL")
MODEL_API_KEY = os.getenv("MODEL_API_KEY")
MODEL = os.getenv("MODEL")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
