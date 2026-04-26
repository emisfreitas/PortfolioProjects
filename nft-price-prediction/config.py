import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("MORALIS_API_KEY")
BASE_URL = "https://deep-index.moralis.io/api/v2.2"
HEADERS = {"X-API-Key": API_KEY}
CHAIN = "eth"