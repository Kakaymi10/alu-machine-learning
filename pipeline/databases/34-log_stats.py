#!/usr/bin/env python3
from pymongo import MongoClient

def log_stats():
    # Connect to the MongoDB server
    client = MongoClient('mongodb://localhost:27017/')
    
    # Select the logs database and nginx collection
    db = client.logs
    collection = db.nginx
    
    # Get the total number of logs
    total_logs = collection.count_documents({})
    print(f"{total_logs} logs")
    
    # Count the number of documents for each method
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    print("Methods:")
    for method in methods:
        count = collection.count_documents({"method": method})
        print(f"\tmethod {method}: {count}")
    
    # Count the number of documents with method=GET and path=/status
    status_check = collection.count_documents({"method": "GET", "path": "/status"})
    print(f"{status_check} status check")

if __name__ == "__main__":
    log_stats()

