import os
import re
import glob
from pathlib import Path

def parse_fasta_file(fasta_path):
    """
    Parse a single FASTA file to extract the input sequence length and metrics for 16 samples
    
    Returns:
    - input_length: Length of the input sequence
    - avg_metrics: Average metrics for 16 samples {perplexity, recovery, edit_dist, sc_score}
    """
    try:
        with open(fasta_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split different sequence blocks
        sequences = content.strip().split('>')
        sequences = [seq.strip() for seq in sequences if seq.strip()]
        
        input_length = 0
        sample_metrics = []
        
        for seq_block in sequences:
            lines = seq_block.strip().split('\n')
            if not lines:
                continue
            
            header = lines[0]
            sequence_lines = lines[1:]
            
            # Process input sequence
            if 'input_sequence' in header:
                # Combine all sequence lines and calculate length
                sequence = ''.join(sequence_lines).replace(' ', '').replace('\n', '')
                input_length = len(sequence)
                print(f"  Input sequence length: {input_length}")
            
            # Process sample sequence
            elif 'sample=' in header:
                # Extract metrics using regular expressions
                perplexity_match = re.search(r'perplexity=([0-9.]+)', header)
                recovery_match = re.search(r'recovery=([0-9.]+)', header)
                edit_dist_match = re.search(r'edit_dist=([0-9.]+)', header)
                sc_score_match = re.search(r'sc_score=([0-9.]+)', header)
                
                if all([perplexity_match, recovery_match, edit_dist_match, sc_score_match]):
                    metrics = {
                        'perplexity': float(perplexity_match.group(1)),
                        'recovery': float(recovery_match.group(1)),
                        'edit_dist': float(edit_dist_match.group(1)),
                        'sc_score': float(sc_score_match.group(1))
                    }
                    sample_metrics.append(metrics)
        
        # Calculate averages
        if sample_metrics:
            avg_metrics = {
                'perplexity': sum(m['perplexity'] for m in sample_metrics) / len(sample_metrics),
                'recovery': sum(m['recovery'] for m in sample_metrics) / len(sample_metrics),
                'edit_dist': sum(m['edit_dist'] for m in sample_metrics) / len(sample_metrics),
                'sc_score': sum(m['sc_score'] for m in sample_metrics) / len(sample_metrics)
            }
            print(f"  Found {len(sample_metrics)} samples")
            print(f"  Average metrics: perplexity={avg_metrics['perplexity']:.4f}, recovery={avg_metrics['recovery']:.4f}, edit_dist={avg_metrics['edit_dist']:.4f}, sc_score={avg_metrics['sc_score']:.4f}")
        else:
            print("  Warning: No valid sample metrics found")
            avg_metrics = None
        
        return input_length, avg_metrics
        
    except Exception as e:
        print(f"  Error: Exception occurred while processing file: {e}")
        return 0, None

def process_all_fasta_files(input_dir="ls216seed1", output_file="data/plotdata/ls216seed1"):
    """
    Process all FASTA files in the specified directory
    """
    print("=== FASTA File Processing Script ===")
    
    # Check input directory
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    # Create output directory
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    # Find all FASTA files
    fasta_pattern = os.path.join(input_dir, "*.fasta")
    fasta_files = glob.glob(fasta_pattern)
    
    if not fasta_files:
        print(f"Warning: No .fasta files found in {input_dir}")
        return
    
    print(f"Found {len(fasta_files)} FASTA files")
    
    # Store processing results
    results = []
    processed_count = 0
    failed_count = 0
    
    # Process each FASTA file
    for i, fasta_file in enumerate(sorted(fasta_files), 1):
        filename = os.path.basename(fasta_file)
        print(f"\n[{i}/{len(fasta_files)}] Processing file: {filename}")
        
        input_length, avg_metrics = parse_fasta_file(fasta_file)
        
        if input_length > 0 and avg_metrics is not None:
            # Format result
            result_line = f"{input_length} {avg_metrics['perplexity']:.4f} {avg_metrics['recovery']:.4f} {avg_metrics['edit_dist']:.4f} {avg_metrics['sc_score']:.4f}"
            results.append(result_line)
            processed_count += 1
            print(f"  ✓ Processed successfully")
        else:
            print(f"  ✗ Processing failed")
            failed_count += 1
    
    # Save results
    if results:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(result + '\n')
            
            print(f"\n=== Processing Completed ===")
            print(f"Total files: {len(fasta_files)}")
            print(f"Successfully processed: {processed_count}")
            print(f"Failed: {failed_count}")
            print(f"Results saved to: {output_file}")
            
            # Show preview of first few lines
            print(f"\nResults preview (first 5 lines):")
            for i, result in enumerate(results[:5]):
                print(f"  {result}")
            if len(results) > 5:
                print(f"  ... (total {len(results)} lines)")
                
        except Exception as e:
            print(f"Error saving file: {e}")
    else:
        print("No successfully processed data, unable to generate output file")

def validate_output_format(output_file):
    """
    Validate the format of the output file
    """
    if not os.path.exists(output_file):
        print("Output file does not exist")
        return
    
    print(f"\n=== Validate Output Format ===")
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"Total lines: {len(lines)}")
        
        for i, line in enumerate(lines[:3], 1):  # Check first 3 lines
            parts = line.strip().split()
            if len(parts) == 5:
                length = int(parts[0])
                perplexity = float(parts[1])
                recovery = float(parts[2])
                edit_dist = float(parts[3])
                sc_score = float(parts[4])
                print(f"Line {i}: length={length}, perplexity={perplexity}, recovery={recovery}, edit_dist={edit_dist}, sc_score={sc_score}")
            else:
                print(f"Line {i} format error: {line.strip()}")
    
    except Exception as e:
        print(f"Error during validation: {e}")

def main():
    """
    Main function
    """
    # Set input and output paths
    input_directory = "ls216seed1"
    output_filepath = "data/plotdata/ls216seed1"
    
    # Process all FASTA files
    process_all_fasta_files(input_directory, output_filepath)
    
    # Validate output format
    validate_output_format(output_filepath)

if __name__ == "__main__":
    main()