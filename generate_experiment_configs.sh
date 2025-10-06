#!/bin/bash

# Script to generate YAML configuration files for RAG experiments
# Usage: ./generate_experiment_configs.sh method1 method2 method3 ...
# Example: ./generate_experiment_configs.sh simple fusion hyde self

# Remove set -e to prevent script from stopping on non-critical errors
# set -e

# Configuration constants
DATASETS_DIR="./datasets"
EXPERIMENTS_DIR="./configs_3b"
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
ALPHA="0.1"
METHOD="attack" # normal, aware, attack, defense
JUST_MODEL_NAME=${MODEL_NAME##*/}
JUST_MODEL_NAME=${JUST_MODEL_NAME,,}
JUST_MODEL_NAME=${JUST_MODEL_NAME//-/_}

# Function to display usage
usage() {
    echo "Usage: $0 <method1> <method2> ... <methodN>"
    echo ""
    echo "Available methods: simple, fusion, hyde, self, graph"
    echo ""
    echo "Each method (except graph) will generate both LLM and RAG versions:"
    echo "  - simple → simplellm, simplerag"
    echo "  - fusion → fusionllm, fusionrag"
    echo "  - hyde → hydellm, hyderag"
    echo "  - self → selfllm, selfrag"
    echo "  - graph → graphrag (special case)"
    echo ""
    echo "Example: $0 simple fusion hyde"
    echo "This will generate configs for: simplellm, simplerag, fusionllm, fusionrag, hydellm, hyderag"
    echo ""
    echo "Generated configs will be saved in: $EXPERIMENTS_DIR/"
    exit 1
}

# Function to convert dataset filename to config name format
dataset_to_config_name() {
    local dataset_file="$1"
    # Remove .json extension and replace underscores with underscores (keep as is)
    echo "${dataset_file%.json}"
}

# Function to convert dataset filename to output name format  
dataset_to_output_name() {
    local dataset_file="$1"
    # Remove .json extension and convert to lowercase
    echo "${dataset_file%.json}" | tr '[:upper:]' '[:lower:]'
}

# Function to generate config file
generate_config() {
    local method_name="$1"
    local dataset_file="$2"
    local output_dir="$3"
    
    local dataset_name=$(dataset_to_config_name "$dataset_file")
    local output_name=$(dataset_to_output_name "$dataset_file")
    local config_filename="${method_name}_${METHOD}_${dataset_name}.yaml"
    local config_path="$output_dir/$config_filename"
    
    # Create the YAML content
    cat > "$config_path" << EOF
system:
  name: $method_name
  alpha: $ALPHA
  args:
    model_name: $MODEL_NAME
    method: $METHOD

dataset: datasets/$dataset_file
output: results/${method_name}/${METHOD}/${JUST_MODEL_NAME}/${output_name}_${ALPHA}
EOF

    echo "Generated: $config_path"
}

# Check if arguments provided
if [ $# -eq 0 ]; then
    echo "Error: No methods specified."
    echo ""
    usage
fi

# Check if datasets directory exists
if [ ! -d "$DATASETS_DIR" ]; then
    echo "Error: Datasets directory '$DATASETS_DIR' not found."
    echo "Please run this script from the URAG root directory."
    exit 1
fi

# Create experiments directory if it doesn't exist
mkdir -p "$EXPERIMENTS_DIR"

# Get list of all JSON datasets
echo "Scanning for datasets in $DATASETS_DIR..."
datasets=($(find "$DATASETS_DIR" -name "*.json" -type f -exec basename {} \; | sort))

if [ ${#datasets[@]} -eq 0 ]; then
    echo "Error: No JSON dataset files found in $DATASETS_DIR"
    exit 1
fi

echo "Found ${#datasets[@]} datasets:"
for dataset in "${datasets[@]}"; do
    echo "  - $dataset"
done
echo ""

# Process each method
methods_to_generate=()
for method in "$@"; do
    methods_to_generate+=("${method}llm" "${method}rag")
done

echo "Will generate configs for methods: ${methods_to_generate[*]}"
echo ""

# Generate configuration files
total_configs=0
for method in "${methods_to_generate[@]}"; do
    echo "Generating configs for method: $method"
    for dataset in "${datasets[@]}"; do
        generate_config "$method" "$dataset" "$EXPERIMENTS_DIR"
        ((total_configs++))
    done
    echo ""
done

echo "================================"
echo "Generation complete!"
echo "Total configs generated: $total_configs"
echo "Configs saved in: $EXPERIMENTS_DIR/"
echo "================================"

# List generated files
echo ""
echo "Generated config files:"
ls -la "$EXPERIMENTS_DIR"/*.yaml 2>/dev/null | awk '{print "  " $9}' || echo "  (No files found)"
