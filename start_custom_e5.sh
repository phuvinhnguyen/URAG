#!/bin/bash
# Start E5 search server with YOUR custom corpus

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUSTOM_CORPUS_DIR="${SCRIPT_DIR}/data/custom_corpus"
EMBEDDINGS_DIR="${CUSTOM_CORPUS_DIR}/embeddings"
CORPUS_FILE="${CUSTOM_CORPUS_DIR}/corpus.jsonl"

echo "=================================================="
echo "Starting E5 Server with Custom Corpus"
echo "=================================================="

# Check if embeddings exist
if [ ! -d "$EMBEDDINGS_DIR" ]; then
    echo "❌ Error: Embeddings not found at $EMBEDDINGS_DIR"
    echo ""
    echo "Run first: python generate_e5_embeddings.py"
    exit 1
fi

# Check if corpus exists
if [ ! -f "$CORPUS_FILE" ]; then
    echo "❌ Error: Corpus not found at $CORPUS_FILE"
    echo ""
    echo "Run first: python generate_e5_embeddings.py"
    exit 1
fi

# Count shards
SHARD_COUNT=$(ls -1 "$EMBEDDINGS_DIR"/shard_*.pt 2>/dev/null | wc -l)
echo "✅ Found $SHARD_COUNT embedding shards"

# Count documents
DOC_COUNT=$(wc -l < "$CORPUS_FILE")
echo "✅ Found $DOC_COUNT documents in corpus"

echo ""
echo "Starting E5 server on http://localhost:8090"
echo "Press Ctrl+C to stop"
echo ""

# Start server
cd "$SCRIPT_DIR"
PYTHONPATH=corag/src python corag/src/search/start_e5_custom.py \
    --index-dir "$EMBEDDINGS_DIR" \
    --corpus "$CORPUS_FILE" \
    --model intfloat/e5-large-v2 \
    --port 8090 \
    --host localhost


