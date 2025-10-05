import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "devsecret")
    MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/stockapp")
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "static/uploads")
    AI_PROVIDER = os.environ.get("AI_PROVIDER", "none").lower()
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "text-bison-001")
    NEWS_API_KEY = "ce7e221c9cb3437688c81fcee4f893c5"

