import os

import pytz
from dotenv import load_dotenv

load_dotenv(override=True)


class CoreCFG:
    PROJECT_NAME = "PolygonAI-GPT-Agent"
    SYSTEM_OS_PATH = str(os.environ["SYSTEM_OS_PATH"])
    SANDBOX_IMAGE = str(os.environ["SANDBOX_IMAGE"])
    OUTPUT_DIR = "data"
    TIME_ZONE = pytz.timezone("Asia/Ho_Chi_Minh")


class AzureOpenAICFG:
    ENDPOINT = str(os.environ["AZURE_OPENAI_ENDPOINT"])
    API_KEY = str(os.environ["AZURE_OPENAI_API_KEY"])
    API_VERSION = str(os.environ["AZURE_OPENAI_API_VERSION"])
    API_DEPLOYMENT = str(os.environ["AZURE_OPENAI_API_DEPLOYMENT"])
    MAX_RETRIES = 3


class AzureBlobStorageCFG:
    AZURE_BLOB_CONNECTION_STRING = str(os.environ["AZURE_BLOB_CONNECTION_STRING"])
    CONTAINER_NAME = str(os.environ["CONTAINER_NAME"])


class MongoDBCFG:
    URI = str(os.environ["MONGODB_URI"])
    DB_NAME = str(os.environ["MONGODB_DB_NAME"])


class QueryVideosCFG:
    ENDPOINTS = os.getenv("QUERY_VIDEOS_URL", "")
    NUM_RESULTS = os.getenv("NUM_RESULTS", "5")
    PROMPT = ""
    START_TIME = ""
    END_TIME = ""
    URL = f"{ENDPOINTS}?prompt={PROMPT}&start_time={START_TIME}&end_time={END_TIME}&num_results={NUM_RESULTS}"
