import json
import argparse
import os

def split_json_to_files(json_filepath, input_text_file, input_tokens_file, output_text_file, output_tokens_file):
    """
    Reads a JSON file and splits data from four fixed fields into four user-specified text files.
    """
    
    # Fixed mapping from the JSON field key to the specific output file path variable
    FIELD_FILE_MAP = {
        'input_text': input_text_file,
        'input_token_ids': input_tokens_file,
        'output_text': output_text_file,
        'output_token_ids': output_tokens_file
    }
    
    # Initialize dictionaries to hold the collected data, using the file path as the key
    collected_data = {filepath: [] for filepath in FIELD_FILE_MAP.values()}

    # --- 1. Read the JSON data ---
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"Error: JSON file '{json_filepath}' does not contain a list of objects (a list of records).")
            return

    except FileNotFoundError:
        print(f"Error: The input JSON file '{json_filepath}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_filepath}'. Check file format.")
        return
    
    print(f"Successfully loaded {len(data)} records from '{json_filepath}'.")

    # --- 2. Iterate and Collect Data ---
    missing_fields_warning = set()

    for record in data:
        for field_key, file_path in FIELD_FILE_MAP.items():
            if field_key in record:
                value = record[field_key]
                if isinstance(value, list):
                    # For token IDs, convert list of numbers to a space-separated string
                    line = ' '.join(map(str, value))
                else:
                    # For text, just ensure it's a string
                    line = str(value)
                
                collected_data[file_path].append(line)
            else:
                # Handle records missing a key
                missing_fields_warning.add(field_key)
                collected_data[file_path].append('') # Append empty line to maintain record alignment

    if missing_fields_warning:
        print(f"\nWarning: The following fields were missing in some records: {', '.join(missing_fields_warning)}")

    # --- 3. Write Data to Files ---
    print("\nWriting collected data to files...")
    for file_path, lines in collected_data.items():
        try:
            with open(file_path, 'w', encoding='utf-8') as outfile:
                outfile.write('\n'.join(lines))
            print(f"✅ Successfully created: {file_path} with {len(lines)} lines.")
        except IOError as e:
            print(f"❌ Error writing to file {file_path}: {e}")
            
    print("\nProcess complete.")

# ----------------- Main Function with Argument Parsing -----------------

def main():
    """Sets up the argument parser for 5 required file paths."""
    parser = argparse.ArgumentParser(
        description="Split a JSON file (list of objects) into four separate text files based on fixed field names, using 5 command-line arguments."
    )
    
    # 1. Input JSON File
    parser.add_argument(
        '--input_json', 
        type=str, 
        help='The path to the input JSON file containing the list of records.'
    )
    
    # 2. Output File for "input_text"
    parser.add_argument(
        '--input_text_file', 
        type=str, 
        help='The path/name for the output file for the "input_text" field.'
    )
    
    # 3. Output File for "input_token_ids"
    parser.add_argument(
        '--input_tokens_file', 
        type=str, 
        help='The path/name for the output file for the "input_token_ids" field.'
    )
    
    # 4. Output File for "output_text"
    parser.add_argument(
        '--output_text_file', 
        type=str, 
        help='The path/name for the output file for the "output_text" field.'
    )
    
    # 5. Output File for "output_token_ids"
    parser.add_argument(
        '--output_tokens_file', 
        type=str, 
        help='The path/name for the output file for the "output_token_ids" field.'
    )
    
    args = parser.parse_args()
    
    # Pass all 5 arguments to the processing function
    split_json_to_files(
        args.input_json, 
        args.input_text_file, 
        args.input_tokens_file, 
        args.output_text_file, 
        args.output_tokens_file
    )

if __name__ == '__main__':
    main()