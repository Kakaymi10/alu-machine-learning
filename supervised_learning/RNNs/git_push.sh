#!/bin/bash

# Prompt for the commit message
echo "Enter your commit message: "
read commit_message

# Stage all changes
git add .

# Commit with the provided message
git commit -m "$commit_message"

# Push the changes
git push

# Success message
echo "Changes have been pushed successfully!"
