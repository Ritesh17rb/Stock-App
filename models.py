# models.py
from pymongo import MongoClient
from bson.objectid import ObjectId
from config import Config
import os

client = MongoClient(Config.MONGO_URI)
db = client.get_default_database()

users = db.users
portfolios = db.portfolios

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

def get_portfolio(user_id):
    try:
        return portfolios.find_one({"user_id": ObjectId(user_id)})
    except Exception:
        return None

def upsert_portfolio(user_id, doc):
    return portfolios.update_one({"user_id": ObjectId(user_id)}, {"$set": doc}, upsert=True)
