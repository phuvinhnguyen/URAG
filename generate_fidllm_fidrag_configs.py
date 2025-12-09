#!/usr/bin/env python3
"""
Script to generate complete fidllm and fidrag configurations for all datasets.
Based on the template configs provided.
"""

import os
import json
from pathlib import Path

def get_dataset_files():
    """Get all available dataset files from the datasets directory."""
    datasets_dir = Path("datasets")
    dataset_files = []
    
    # Get all .json files in the datasets directory
    for file_path in datasets_dir.glob("*.json"):
        dataset_files.append(file_path.name)
    
    return sorted(dataset_files)

def get_dataset_name_from_file(filename):
    """Extract clean dataset name from filename for use in config names."""
    # Remove .json extension
    name = filename.replace('.json', '')
    return name

def generate_fidllm_config(dataset_file,method, alpha=0.1):
    """Generate fidllm config for a given dataset."""
    dataset_name = get_dataset_name_from_file(dataset_file)
    
    config_content = f"""system:
  name: fidllm
  alpha: {alpha}
  args:
    model_name: google/flan-t5-base
    fid_model_name: Intel/fid_flan_t5_base_nq
    method: {method}

dataset: datasets/{dataset_file}
output: results/fidllm/{method}/flan_t5_base/{dataset_name}_{alpha}
"""
    return config_content

def generate_fidrag_config(dataset_file, method, alpha=0.1, num_samples=20):
    """Generate fidrag config for a given dataset."""
    dataset_name = get_dataset_name_from_file(dataset_file)
    
    config_content = f"""system:
  name: fidrag
  alpha: {alpha}
  args:
    model_name: google/flan-t5-base
    fid_model_name: Intel/fid_flan_t5_base_nq
    method: {method}

dataset: datasets/{dataset_file}
output: results/fidrag/{method}/flan_t5_base/{dataset_name}_{alpha}
"""
    return config_content

def main():
    """Main function to generate all configs."""
    # Create configs directory if it doesn't exist
    configs_dir = Path("configs_8b")
    configs_dir.mkdir(exist_ok=True)
    
    # Get all dataset files
    dataset_files = get_dataset_files()
    
    print(f"Found {len(dataset_files)} dataset files:")
    for dataset_file in dataset_files:
        print(f"  - {dataset_file}")
    
    print("\nGenerating fidllm and fidrag configs...")
    
    generated_configs = []
    
    for dataset_file in dataset_files:
        dataset_name = get_dataset_name_from_file(dataset_file)
        
        # Generate fidllm normal config
        fidllm_config_name_normal = f"fidllm_normal_{dataset_name}.yaml"
        fidllm_config_path_normal = configs_dir / fidllm_config_name_normal
        #attack
        fidllm_config_name_attack = f"fidllm_attack_{dataset_name}.yaml"
        fidllm_config_path_attack = configs_dir / fidllm_config_name_attack
        #aware
        fidllm_config_name_aware = f"fidllm_aware_{dataset_name}.yaml"
        fidllm_config_path_aware = configs_dir / fidllm_config_name_aware
        # Only create if it doesn't exist
        if not fidllm_config_path_normal.exists():
            fidllm_content = generate_fidllm_config(dataset_file, method="normal")
            with open(fidllm_config_path_normal, 'w', encoding='utf-8') as f:
                f.write(fidllm_content)
            generated_configs.append(fidllm_config_name_normal)
      
        else:
            print(f"- Skipped (exists): {fidllm_config_name_normal}")
        if not fidllm_config_path_attack.exists():
            fidllm_content = generate_fidllm_config(dataset_file, method="attack")
            with open(fidllm_config_path_attack, 'w', encoding='utf-8') as f:
                f.write(fidllm_content)
            generated_configs.append(fidllm_config_name_attack)
        else:
            print(f"- Skipped (exists): {fidllm_config_name_attack}")
        if not fidllm_config_path_aware.exists():
            fidllm_content = generate_fidllm_config(dataset_file, method="aware")
            with open(fidllm_config_path_aware, 'w', encoding='utf-8') as f:
                f.write(fidllm_content)
            generated_configs.append(fidllm_config_name_aware)
        else:
            print(f"- Skipped (exists): {fidllm_config_name_aware}")
        # Generate fidrag config
        fidrag_config_name_normal = f"fidrag_normal_{dataset_name}.yaml"
        fidrag_config_path_normal = configs_dir / fidrag_config_name_normal
        #attack
        fidrag_config_name_attack = f"fidrag_attack_{dataset_name}.yaml"
        fidrag_config_path_attack = configs_dir / fidrag_config_name_attack
        #aware
        fidrag_config_name_aware = f"fidrag_aware_{dataset_name}.yaml"
        fidrag_config_path_aware = configs_dir / fidrag_config_name_aware
        # Generate fidrag config
        # Only create if it doesn't exist
        if not fidrag_config_path_normal.exists():
            fidrag_content = generate_fidrag_config(dataset_file, method="normal")
            with open(fidrag_config_path_normal, 'w', encoding='utf-8') as f:
                f.write(fidrag_content)
            generated_configs.append(fidrag_config_path_normal)
            print(f"✓ Generated: {fidrag_config_path_normal}")
        else:
            print(f"- Skipped (exists): {fidrag_config_path_normal}")
        if not fidrag_config_path_attack.exists():
            fidrag_content = generate_fidrag_config(dataset_file, method="attack")
            with open(fidrag_config_path_attack, 'w', encoding='utf-8') as f:
                f.write(fidrag_content)
            generated_configs.append(fidrag_config_name_attack)
        else:
            print(f"- Skipped (exists): {fidrag_config_name_attack}")
        if not fidrag_config_path_aware.exists():
            fidrag_content = generate_fidrag_config(dataset_file, method="aware")
            with open(fidrag_config_path_aware, 'w', encoding='utf-8') as f:
                f.write(fidrag_content)
            generated_configs.append(fidrag_config_name_aware)
        else:
            print(f"- Skipped (exists): {fidrag_config_name_aware}")

    
    print(f"\n✅ Complete! Generated {len(generated_configs)} new config files:")
    for config in generated_configs:
        print(f"  - {config}")
    
    if not generated_configs:
        print("All configs already exist!")

if __name__ == "__main__":
    main()
