#!/bin/bash
#
# Script to launch KùzuExplorer for visualizing the Axon knowledge graph
#
# Usage:
#   ./scripts/visualize-graph.sh          # From the project root (where .axon exists)
#   ./scripts/visualize-graph.sh /path/to/project  # Specify project path
#

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${1:-.}"

# Resolve absolute path
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"

AXON_DIR="$PROJECT_DIR/.axon"

echo "🔍 Axon Graph Visualizer"
echo "========================"
echo "Project directory: $PROJECT_DIR"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed."
    echo "   Please install Docker from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Error: Docker is not running."
    echo "   Please start Docker Desktop and try again."
    exit 1
fi

# Check if .axon/kuzu file or directory exists
if [ ! -e "$AXON_DIR/kuzu" ]; then
    echo "❌ Error: No Axon index found at $AXON_DIR/kuzu"
    echo ""
    echo "   Run 'axon analyze .' first to index your codebase."
    exit 1
fi

echo "✅ Docker is running"
echo "✅ Axon index found at $AXON_DIR/kuzu"
echo ""

# Kill any existing container on port 8000
EXISTING_CONTAINER=$(docker ps -q --filter "publish=8000")
if [ -n "$EXISTING_CONTAINER" ]; then
    echo "🧹 Removing existing container on port 8000..."
    docker kill $EXISTING_CONTAINER 2>/dev/null || true
fi

echo "🚀 Starting KùzuExplorer..."
echo "   URL: http://localhost:8000"
echo ""
echo "   Press Ctrl+C to stop the server"
echo ""

# Run KùzuExplorer with Docker
docker run -p 8000:8000 \
    -v "$AXON_DIR:/database" \
    -e KUZU_FILE=kuzu \
    -e MODE=READ_ONLY \
    --rm \
    kuzudb/explorer:latest
