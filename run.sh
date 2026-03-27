for cfg in configs/*/*.yaml; do
    output_path=$(cat "$cfg" | grep "^output:" | sed 's/^output: //' | sed 's/^[[:space:]]*//')
    
    if [ ! -f "$output_path/evaluation_metrics.json" ]; then
        python cli.py --config "$cfg"
    fi
done