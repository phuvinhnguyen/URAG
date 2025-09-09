#!/bin/bash

# Script to compare RAG vs LLM performance for all matching method pairs
# This script automatically finds all RAG/LLM pairs in the results folder
# and runs compare_performance.py on their evaluation_metrics.json files

set -e

# Configuration constants
RESULTS_DIR="./results"
COMPARISONS_DIR="./comparisons"
METRICS_FILE="evaluation_metrics.json"
COMPARE_SCRIPT="./compare_performance.py"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "This script compares RAG vs LLM performance for all matching method pairs"
    echo "found in the results directory. It looks for pairs like:"
    echo "  - fusionrag_* vs fusionllm_*"
    echo "  - simplerag_* vs simplellm_*"
    echo "  - hyderag_* vs hydellm_*"
    echo "  - selfrag_* vs selfllm_*"
    echo ""
    echo "Options:"
    echo "  --dry-run        Show what would be compared without running comparisons"
    echo "  --quiet          Suppress detailed output during comparison"
    echo "  --force          Overwrite existing comparison files"
    echo "  --help           Show this help message"
    echo ""
    echo "Output:"
    echo "  Comparison results will be saved in: $COMPARISONS_DIR/"
    echo "  Each comparison will be named: {method}_{dataset}_rag_vs_llm_comparison.json"
    echo ""
    echo "Requirements:"
    echo "  - Python script '$COMPARE_SCRIPT' must exist"
    echo "  - Results directory '$RESULTS_DIR' must exist"
    echo "  - Each result directory must contain '$METRICS_FILE'"
    exit 0
}

# Function to log messages
log() {
    if [ "$QUIET" != "true" ]; then
        echo "$1"
    fi
}

# Function to extract method name from directory name
extract_method() {
    local dir_name="$1"
    # Extract just the base method name (simple, fusion, hyde, self, etc.)
    echo "$dir_name" | sed -E 's/(llm|rag)_.*$//'
}

# Function to extract dataset name from directory name
extract_dataset() {
    local dir_name="$1"
    # Extract everything after the model name using simpler pattern matching
    if [[ "$dir_name" == *"_llama_3.1_8b_instruct_"* ]]; then
        # Split on the pattern and take the last part
        echo "${dir_name##*_llama_3.1_8b_instruct_}"
    elif [[ "$dir_name" == *"_qwen3_0.6b_"* ]]; then
        echo "${dir_name##*_qwen3_0.6b_}"
    else
        # Fallback: extract everything after the third underscore
        echo "$dir_name" | cut -d'_' -f4-
    fi
}

# Function to generate comparison filename
generate_comparison_filename() {
    local method="$1"
    local dataset="$2"
    echo "${method}_${dataset}_rag_vs_llm_comparison.json"
}

# Function to check if directories and files exist
validate_pair() {
    local llm_dir="$1"
    local rag_dir="$2"
    
    if [ ! -d "$RESULTS_DIR/$llm_dir" ]; then
        log "Warning: LLM directory not found: $RESULTS_DIR/$llm_dir"
        return 1
    fi
    
    if [ ! -d "$RESULTS_DIR/$rag_dir" ]; then
        log "Warning: RAG directory not found: $RESULTS_DIR/$rag_dir"
        return 1
    fi
    
    if [ ! -f "$RESULTS_DIR/$llm_dir/$METRICS_FILE" ]; then
        log "Warning: LLM metrics file not found: $RESULTS_DIR/$llm_dir/$METRICS_FILE"
        return 1
    fi
    
    if [ ! -f "$RESULTS_DIR/$rag_dir/$METRICS_FILE" ]; then
        log "Warning: RAG metrics file not found: $RESULTS_DIR/$rag_dir/$METRICS_FILE"
        return 1
    fi
    
    return 0
}

# Function to run comparison
run_comparison() {
    local llm_dir="$1"
    local rag_dir="$2"
    local method="$3"
    local dataset="$4"
    
    local llm_metrics="$RESULTS_DIR/$llm_dir/$METRICS_FILE"
    local rag_metrics="$RESULTS_DIR/$rag_dir/$METRICS_FILE"
    local comparison_file="$COMPARISONS_DIR/$(generate_comparison_filename "$method" "$dataset")"
    
    # Check if output file already exists
    if [ -f "$comparison_file" ] && [ "$FORCE" != "true" ]; then
        log "Skipping $method/$dataset (comparison file already exists: $comparison_file)"
        return 0
    fi
    
    log "Comparing: $method/$dataset"
    log "  RAG:  $rag_metrics"
    log "  LLM:  $llm_metrics"
    log "  Output: $comparison_file"
    
    # Run the comparison (RAG as file1, LLM as file2, so ratio > 1 means RAG is better)
    if [ "$QUIET" = "true" ]; then
        python "$COMPARE_SCRIPT" "$rag_metrics" "$llm_metrics" "$comparison_file" --quiet
    else
        python "$COMPARE_SCRIPT" "$rag_metrics" "$llm_metrics" "$comparison_file"
    fi
    
    log "✓ Comparison completed for $method/$dataset"
    log ""
}

# Parse command line arguments
DRY_RUN=false
QUIET=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --quiet)
            QUIET=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate prerequisites
if [ ! -f "$COMPARE_SCRIPT" ]; then
    echo "Error: Compare script not found: $COMPARE_SCRIPT"
    echo "Please make sure the compare_performance.py script exists in the current directory."
    exit 1
fi

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory not found: $RESULTS_DIR"
    echo "Please run this script from the URAG root directory."
    exit 1
fi

# Create comparisons directory if it doesn't exist
mkdir -p "$COMPARISONS_DIR"

log "Scanning for RAG/LLM pairs in $RESULTS_DIR..."

# Get all LLM directories
llm_dirs=($(ls "$RESULTS_DIR" | grep "llm_" | sort))
log "Found ${#llm_dirs[@]} LLM result directories"

# Find matching RAG directories for each LLM directory
pairs_found=0
pairs_processed=0
pairs_skipped=0

for llm_dir in "${llm_dirs[@]}"; do
    # Generate corresponding RAG directory name
    rag_dir=$(echo "$llm_dir" | sed 's/llm_/rag_/')
    
    # Extract method and dataset
    method=$(extract_method "$llm_dir")
    dataset=$(extract_dataset "$llm_dir")
    
    # Skip if we can't extract proper method/dataset
    if [ -z "$method" ] || [ -z "$dataset" ]; then
        log "Warning: Could not extract method/dataset from: $llm_dir"
        continue
    fi
    
    # Check if RAG directory exists and both have metrics files
    if validate_pair "$llm_dir" "$rag_dir"; then
        pairs_found=$((pairs_found + 1))
        
        if [ "$DRY_RUN" = "true" ]; then
            log "Would compare: $method on $dataset"
            log "  RAG:  $RESULTS_DIR/$rag_dir/$METRICS_FILE"
            log "  LLM:  $RESULTS_DIR/$llm_dir/$METRICS_FILE"
            log "  Output: $COMPARISONS_DIR/$(generate_comparison_filename "$method" "$dataset")"
            log ""
        else
            # Check if comparison already exists
            comparison_file="$COMPARISONS_DIR/$(generate_comparison_filename "$method" "$dataset")"
            if [ -f "$comparison_file" ] && [ "$FORCE" != "true" ]; then
                pairs_skipped=$((pairs_skipped + 1))
                log "Skipping $method/$dataset (already exists: $comparison_file)"
            else
                run_comparison "$llm_dir" "$rag_dir" "$method" "$dataset"
                pairs_processed=$((pairs_processed + 1))
            fi
        fi
    fi
done

# Print summary
echo "================================"
if [ "$DRY_RUN" = "true" ]; then
    echo "DRY RUN SUMMARY"
    echo "Found $pairs_found RAG/LLM pairs that would be compared"
else
    echo "COMPARISON SUMMARY"
    echo "Found: $pairs_found RAG/LLM pairs"
    echo "Processed: $pairs_processed comparisons"
    echo "Skipped: $pairs_skipped (already existed)"
    echo "Results saved in: $COMPARISONS_DIR/"
fi
echo "================================"

# List generated files if not dry run
if [ "$DRY_RUN" != "true" ] && [ $pairs_processed -gt 0 ]; then
    echo ""
    echo "Generated comparison files:"
    ls -la "$COMPARISONS_DIR"/*.json 2>/dev/null | awk '{print "  " $9}' || echo "  (No files found)"
fi
