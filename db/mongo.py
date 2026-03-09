from pymongo import MongoClient
from typing import Optional
import os

MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "autentica")

_mongo_client: Optional[MongoClient] = None


def get_mongo_client() -> MongoClient:
    global _mongo_client

    if _mongo_client is None:

        if not MONGO_URI:
            raise RuntimeError("MONGO_URI missing")

        _mongo_client = MongoClient(
            MONGO_URI,
            socketTimeoutMS=120000,
            connectTimeoutMS=20000,
            serverSelectionTimeoutMS=20000,
            retryWrites=False
        )

    return _mongo_client


def get_db():

    client = get_mongo_client()

    return client[MONGO_DB_NAME]
