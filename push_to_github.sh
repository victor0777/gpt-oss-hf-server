#!/bin/bash

# GitHub Push Script for GPT-OSS HF Server
# You need to provide your GitHub username and personal access token

echo "=========================================="
echo "GitHub Push Script"
echo "=========================================="

# Check if we're in the right directory
if [ ! -d .git ]; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Option 1: Using GitHub Personal Access Token (Recommended)
echo "Option 1: Using Personal Access Token"
echo "--------------------------------------"
echo "1. Go to GitHub Settings > Developer settings > Personal access tokens"
echo "2. Generate a new token with 'repo' scope"
echo "3. Use the token as your password when prompted"
echo ""
echo "To push with token:"
echo "git push https://<username>:<token>@github.com/victor0777/gpt-oss-hf-server.git main"
echo ""

# Option 2: Using SSH (Alternative)
echo "Option 2: Using SSH Key"
echo "--------------------------------------"
echo "1. Generate SSH key: ssh-keygen -t ed25519 -C 'your_email@example.com'"
echo "2. Add to GitHub: Settings > SSH and GPG keys"
echo "3. Change remote to SSH:"
echo "git remote set-url origin git@github.com:victor0777/gpt-oss-hf-server.git"
echo "git push -u origin main"
echo ""

# Option 3: Using GitHub CLI (if installed)
echo "Option 3: Using GitHub CLI"
echo "--------------------------------------"
echo "1. Install: gh auth login"
echo "2. Push: gh repo create victor0777/gpt-oss-hf-server --public --source=. --push"
echo ""

# Current repository status
echo "Current Repository Status:"
echo "--------------------------------------"
git status --short
echo ""
echo "Remote URL:"
git remote -v
echo ""
echo "Latest commit:"
git log --oneline -1
echo ""

echo "=========================================="
echo "Ready to push. Choose one of the options above."
echo "==========================================">