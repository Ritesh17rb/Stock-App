from pymongo import MongoClient
from bson.objectid import ObjectId
from config import Config
import os
import datetime

client = MongoClient(Config.MONGO_URI)
db = client.get_default_database()

users = db.users
portfolios = db.portfolios
articles_col = db.articles  # new collection

# --- Users ---
def create_user(doc):
    return users.insert_one(doc)

def find_user_by_email(email):
    return users.find_one({"email": email})

def find_user_by_id(_id):
    try:
        return users.find_one({"_id": ObjectId(_id)})
    except Exception:
        return None

def update_user(_id, update_doc):
    return users.update_one({"_id": ObjectId(_id)}, {"$set": update_doc})

def delete_user(_id):
    return users.delete_one({"_id": ObjectId(_id)})

# --- Portfolio ---
def get_portfolio(user_id):
    try:
        return portfolios.find_one({"user_id": ObjectId(user_id)})
    except Exception:
        return None

def upsert_portfolio(user_id, doc):
    return portfolios.update_one({"user_id": ObjectId(user_id)}, {"$set": doc}, upsert=True)

# --- Articles ---
def create_article(user_id, title, content, image_url=None):
    doc = {
        "user_id": user_id,
        "title": title,
        "content": content,
        "image_url": image_url,
        "likes": 0,
        "dislikes": 0,
        "created_at": datetime.datetime.utcnow()
    }
    return articles_col.insert_one(doc)

def get_articles():
    return list(articles_col.find().sort("created_at", -1))

def like_article(article_id):
    return articles_col.update_one({"_id": ObjectId(article_id)}, {"$inc": {"likes": 1}})

def dislike_article(article_id):
    return articles_col.update_one({"_id": ObjectId(article_id)}, {"$inc": {"dislikes": 1}})

def delete_article(article_id, user_id):
    return articles_col.delete_one({"_id": ObjectId(article_id), "user_id": user_id})
# --- Get single article by ID ---
def get_article_by_id(article_id):
    try:
        article = articles_col.find_one({"_id": ObjectId(article_id)})
        if not article:
            return None

        # Convert _id to string for Jinja URLs
        article["_id_str"] = str(article["_id"])

        # Fetch author's name
        user = users.find_one({"_id": ObjectId(article["user_id"])})
        article["author_name"] = user.get("name") if user else "Unknown"

        # Ensure user_id is string
        article["user_id"] = str(article["user_id"])
        return article
    except Exception:
        return None

