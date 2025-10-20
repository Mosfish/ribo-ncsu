#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

def read_test(index_file="test_index.txt"):
    """
    Read test_index.txt file to get the list of PDB filenames (without extension).
    """
    try:
        with open(index_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        pdb_basenames = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                pdb_basenames.append(line)
        
        return pdb_basenames
    
    except FileNotFoundError:
        print(f"Error: File {index_file} not found")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

def run_command(pdb_basename):
    """
    Execute the run.py command for the specified PDB basename.
    """
    # **MODIFIED**: Add the .pdb extension to the basename to form the full filename
    pdb_file = f"{pdb_basename}.pdb"
    output_name = pdb_basename
    
    # Build command
    cmd = [
        "python", "run.py",
        "--pdb_filepath", f"data/raw/{pdb_file}",
        "--output_filepath", f"edges2/{output_name}.fasta",
        "--split", "das",
        "--max_num_conformers", "1",
        "--n_samples", "16",
        "--temperature", "0.5"
    ]
    
    print(f"Processing: {pdb_file}")
    print(f"Executing command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ Successfully processed {pdb_file}")
        print(f"Output file: edges2/{output_name}.fasta")
        
        if result.stdout:
            print("Standard output:")
            print(result.stdout[:200] + ("..." if len(result.stdout) > 200 else ""))
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to process {pdb_file}")
        print(f"Error code: {e.returncode}")
        if e.stderr:
            print(f"Error message: {e.stderr}")
        return False
    
    except Exception as e:
        print(f"✗ Unexpected error occurred while processing {pdb_file}: {e}")
        return False

def main():
    """
    Main function
    """
    print("=== RNA Test Script Started ===")
    
    # Check for required files and directories
    required_files = ["run.py", "test_index.txt"]
    for f in required_files:
        if not os.path.exists(f):
            print(f"Error: Required file '{f}' not found in the current directory")
            sys.exit(1)
    
    # Ensure output directory exists
    output_dir = "edges2"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Read test index file
    pdb_basenames = read_test()
    
    if not pdb_basenames:
        print("Warning: test_index.txt file is empty or failed to read. Exiting.")
        sys.exit(1)
    
    print(f"Found {len(pdb_basenames)} PDB files to process:")
    for i, name in enumerate(pdb_basenames, 1):
        print(f"  {i}. {name}")
    
    print("\nStarting processing...")
    
    success_count = 0
    failed_files = []
    
    # Process each PDB file
    for i, pdb_basename in enumerate(pdb_basenames, 1):
        print(f"\n[{i}/{len(pdb_basenames)}] " + "="*50)
        
        # **MODIFIED**: Construct the full input path with the .pdb extension
        input_filename = f"{pdb_basename}.pdb"
        input_path = Path("data/raw") / input_filename
        
        if not input_path.exists():
            print(f"Warning: Input file {input_path} does not exist, skipping...")
            failed_files.append(pdb_basename)
            continue
        
        # **MODIFIED**: Pass the basename (without extension) to the run_command function
        if run_command(pdb_basename):
            success_count += 1
        else:
            failed_files.append(pdb_basename)
    
    # Print summary
    print("\n" + "="*60)
    print("=== Processing Completed ===")
    print(f"Total: {len(pdb_basenames)} files")
    print(f"Success: {success_count} files")
    print(f"Failed: {len(failed_files)} files")
    
    if failed_files:
        print("\nFailed files (basenames):")
        for failed_file in failed_files:
            print(f"  - {failed_file}")
    
    print(f"\nOutput files saved in: {output_dir}/")

if __name__ == "__main__":
    main()