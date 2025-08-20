# https://github.com/facebookresearch/CRAG
"""
CRAG Dataset Downloader and Converter
Downloads CRAG dataset files, extracts them, and converts to JSON format
"""

import os
import shutil
import json
import jsonlines
import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'jsonlines'])
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        sys.exit(1)

def download_file(url, output_path):
    """Download a single file with error handling"""
    try:
        cmd = ['wget', url, '-O', output_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Downloaded: {os.path.basename(output_path)}")
            return True
        else:
            print(f"✗ Failed to download {url}: {result.stderr}")
            return False
    except FileNotFoundError:
        # Try with curl if wget is not available
        try:
            cmd = ['curl', '-L', url, '-o', output_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ Downloaded: {os.path.basename(output_path)}")
                return True
            else:
                print(f"✗ Failed to download {url}: {result.stderr}")
                return False
        except FileNotFoundError:
            print("✗ Neither wget nor curl is available. Please install one of them.")
            return False

def extract_bz2(file_path):
    """Extract bz2 files"""
    try:
        cmd = ['bunzip2', file_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            extracted_path = file_path.replace('.bz2', '')
            print(f"✓ Extracted: {os.path.basename(extracted_path)}")
            return extracted_path
        else:
            print(f"✗ Failed to extract {file_path}: {result.stderr}")
            return None
    except FileNotFoundError:
        print("✗ bunzip2 not found. Please install bzip2 utilities.")
        return None

def merge_tar_parts(base_path, parts):
    """Merge tar file parts"""
    try:
        output_file = f"{base_path}.tar.bz2"
        with open(output_file, 'wb') as outfile:
            for part_file in parts:
                if os.path.exists(part_file):
                    with open(part_file, 'rb') as infile:
                        outfile.write(infile.read())
                else:
                    print(f"✗ Part file not found: {part_file}")
                    return None
        print(f"✓ Merged tar parts: {os.path.basename(output_file)}")
        return output_file
    except Exception as e:
        print(f"✗ Failed to merge tar parts: {e}")
        return None

def convert_jsonl_to_json(jsonl_path, output_path):
    """Convert JSONL file to structured JSON format"""
    try:
        full_data = []
        with jsonlines.open(jsonl_path) as reader:
            for line in reader:
                full_data.append({
                    'id': line.get('interaction_id', ''),
                    'question': line.get('query', ''),
                    'options': [],  # CRAG doesn't have multiple choice options
                    'correct_answer': line.get('answer', ''),
                    'search_results': line.get('search_results', []),
                })
        
        # Split data into calibration (first half) and test (second half)
        split_point = len(full_data) // 2
        
        data = {
            'name': 'CRAG',
            'description': 'CRAG is a dataset for evaluating the performance of RAG systems.',
            'version': '1.0',
            'total_samples': len(full_data),
            'calibration_samples': split_point,
            'test_samples': len(full_data) - split_point,
            'calibration': full_data[:split_point],
            'test': full_data[split_point:]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Converted: {os.path.basename(output_path)} ({len(full_data)} samples)")
        return True
        
    except Exception as e:
        print(f"✗ Failed to convert {jsonl_path}: {e}")
        return False

def main():
    """Main function to download and process CRAG dataset"""
    
    print("CRAG Dataset Downloader and Converter")
    print("=" * 40)
    
    # Install dependencies
    install_dependencies()
    
    # Define download URLs
    download_urls = [
        'https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_1_and_2_dev_v4.jsonl.bz2',
        # 'https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_1_and_2_dev_v5.jsonl.bz2',
        # # Uncomment these if you want to download task 3 data as well
        # 'https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_3_dev_v4.tar.bz2.part1',
        # 'https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_3_dev_v4.tar.bz2.part2',
        # 'https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_3_dev_v4.tar.bz2.part3',
        # 'https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_3_dev_v4.tar.bz2.part4',
        # 'https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_3_dev_v5.tar.bz2.part0',
        # 'https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_3_dev_v5.tar.bz2.part1',
        # 'https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_3_dev_v5.tar.bz2.part2',
        # 'https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_3_dev_v5.tar.bz2.part3',
    ]
    
    # Create output directory
    crag_dir = Path('crag')
    crag_dir.mkdir(exist_ok=True)
    print(f"✓ Created directory: {crag_dir}")
    
    # Download files
    print("\n📥 Downloading files...")
    downloaded_files = []
    for url in download_urls:
        filename = url.split('/')[-1]
        output_path = crag_dir / filename
        
        if download_file(url, str(output_path)):
            downloaded_files.append(str(output_path))
    
    # Extract compressed files
    print("\n📦 Extracting files...")
    jsonl_files = []
    
    for file_path in downloaded_files:
        if file_path.endswith('.jsonl.bz2'):
            extracted = extract_bz2(file_path)
            if extracted and os.path.exists(extracted):
                jsonl_files.append(extracted)
    
    # Handle tar.bz2 parts (if any were downloaded)
    # This section would handle task 3 data if uncommented above
    
    # Convert JSONL files to JSON format
    print("\n🔄 Converting to JSON format...")
    for jsonl_file in jsonl_files:
        base_name = os.path.basename(jsonl_file).replace('.jsonl', '')
        output_name = f'CRAG_{base_name}.json'
        convert_jsonl_to_json(jsonl_file, output_name)
    
    print("\n✅ Processing complete!")
    print(f"📁 Downloaded files are in: {crag_dir}")
    print(f"📄 JSON datasets are in current directory")

    try:
        shutil.rmtree(crag_dir)
        print("✓ Cleanup completed")
    except Exception as e:
        print(f"✗ Cleanup failed: {e}")

if __name__ == "__main__":
    main()