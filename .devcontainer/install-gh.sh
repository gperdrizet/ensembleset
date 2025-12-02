#!/bin/bash
#
# install-gh.sh
# 
# Purpose: Install GitHub CLI (gh) in the development container
# Usage: Called automatically by VS Code devcontainer's onCreateCommand
# When: Runs once during initial container creation
#
# This script installs the GitHub CLI tool which can be used for:
# - Git credential management
# - GitHub API interactions
# - Repository management from the command line

# Exit immediately if any command fails
set -e

echo "Installing GitHub CLI (gh)..."

# Download and install the GitHub CLI GPG key for package verification
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | \
    sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg

# Set proper permissions on the keyring
sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg

# Add the GitHub CLI repository to apt sources
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | \
    sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null

# Update package lists to include the new repository
sudo apt update

# Install GitHub CLI
sudo apt install gh -y

echo "GitHub CLI installed successfully!"
echo "Run 'gh auth login' to authenticate with GitHub."