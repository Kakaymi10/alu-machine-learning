#!/usr/bin/env python3
"""
This script provides statistics about Nginx logs stored in MongoDB.

It connects to a MongoDB instance, accesses the 'logs' database and the 'nginx' collection,
and prints the following statistics:
1. The total number of logs.
2. The number of logs for each HTTP method (GET, POST, PUT, PATCH, DELETE).
3. The number of logs where the method is GET and the path is /status.
"""

from pymongo import MongoClient

def log_stats():
    """
    Retrieve and print statistics from the 'nginx' collection in the 'logs' database.
    """
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
