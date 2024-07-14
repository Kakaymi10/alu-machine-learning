#!/usr/bin/env python3
"""
A script to generate statistics from the 'nginx' collection in the 'logs' MongoDB database.
It prints:
1. Total number of logs.
2. Number of logs for each HTTP method: GET, POST, PUT, PATCH, DELETE.
3. Number of logs where method is GET and path is /status.
"""

from pymongo import MongoClient

def get_log_stats():
    """
    Connects to MongoDB, retrieves and prints log statistics.
    """
    # Connect to MongoDB and select the 'nginx' collection
    client = MongoClient()
    collection = client.logs.nginx

    # Get total number of logs
    total_logs = collection.count_documents({})

    # Count logs for each HTTP method
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    method_counts = {method: collection.count_documents({"method": method}) for method in methods}

    # Count logs where method is GET and path is /status
    status_check_count = collection.count_documents({"method": "GET", "path": "/status"})

    # Print the results
    print("{} logs".format(total_logs))
    print("Methods:")
    for method in methods:
        print("\tmethod {}: {}".format(method, method_counts[method]))
    print("{} status check".format(status_check_count))

if __name__ == "__main__":
    get_log_stats()

