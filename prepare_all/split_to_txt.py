import json
import argparse
import os

def split_json_to_files(json_filepath, input_text_file, input_tokens_file, output_text_file, output_tokens_file, image_path_file):
    """
    Reads a JSON file and splits data into four user-specified text files. 
    Internal newlines ('\n') in text fields are explicitly escaped to '\\n'.
    """
    
    # Fixed mapping from the JSON field key to the specific output file path variable
    FIELD_FILE_MAP = {
        'input_text': input_text_file,
        'input_token_ids': input_tokens_file,
        'output_text': output_text_file,
        'output_token_ids': output_tokens_file,
        'image': image_path_file
    }
    
    # Initialize dictionaries to hold the collected data
    collected_data = {filepath: [] for filepath in FIELD_FILE_MAP.values()}

    # --- 1. Read the JSON data ---
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"Error: JSON file '{json_filepath}' does not contain a list of objects (a list of records).")
            return

    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        print(f"Error reading/parsing '{json_filepath}': {e}")
        return
    
    print(f"Successfully loaded {len(data)} records from '{json_filepath}'.")
    print("Internal newlines (\\n) in text fields will be converted to the literal string '\\\\n'.")

    # --- 2. Iterate and Collect Data ---
    missing_fields_warning = set()

    for record in data:
        for field_key, file_path in FIELD_FILE_MAP.items():
            if field_key in record:
                value = record[field_key]
                
                if field_key in ['input_text', 'output_text']:
                    # Text fields: Explicitly replace the newline character with the literal string '\n'
                    # The replace function is safe and handles both Python's internal string representation
                    # and the actual character.
                    line = str(value).replace('\n', '\\n')
                
                elif field_key in ['input_token_ids', 'output_token_ids'] and isinstance(value, list):
                    # Token ID lists: convert list of numbers to a space-separated string
                    line = ' '.join(map(str, value))
                
                else:
                    # Fallback
                    line = str(value)
                
                collected_data[file_path].append(line)
            else:
                # Handle records missing a key
                missing_fields_warning.add(field_key)
                # Append empty string to maintain record alignment
                collected_data[file_path].append('') 

    if missing_fields_warning:
        print(f"\nWarning: The following fields were missing in some records: {', '.join(missing_fields_warning)}")

    # --- 3. Write Data to Files ---
    print("\nWriting collected data to files...")
    for file_path, lines in collected_data.items():
        try:
            with open(file_path, 'w', encoding='utf-8') as outfile:
                # Join records using a standard newline. Since internal '\n's were 
                # converted to '\\n', this newline only separates records.
                outfile.write('\n'.join(lines))
            print(f"✅ Successfully created: {file_path} with {len(lines)} records.")
        except IOError as e:
            print(f"❌ Error writing to file {file_path}: {e}")
            
    print("\nProcess complete.")

# ----------------- Main Function with Argument Parsing -----------------

def main():
    """Sets up the argument parser for 5 required file paths."""
    parser = argparse.ArgumentParser(
        description="Split a JSON file into four separate text files. Internal newlines (\\n) in text fields are explicitly escaped to '\\n' to ensure one record per line."
    )
    
    # 1. Input JSON File
    parser.add_argument('--input_json', type=str, help='The path to the input JSON file.')
    
    # 2. Output File for "input_text"
    parser.add_argument('--input_text_file', type=str, help='Output path for "input_text".')
    
    # 3. Output File for "input_token_ids"
    parser.add_argument('--input_tokens_file', type=str, help='Output path for "input_token_ids".')
    
    # 4. Output File for "output_text"
    parser.add_argument('--output_text_file', type=str, help='Output path for "output_text".')
    
    # 5. Output File for "output_token_ids"
    parser.add_argument('--output_tokens_file', type=str, help='Output path for "output_token_ids".')
    
    # 5. Output File for "output_token_ids"
    parser.add_argument('--image_path_file', type=str, help='Output path for "image".')
    
    args = parser.parse_args()
    
    split_json_to_files(
        args.input_json,
        args.input_text_file,
        args.input_tokens_file,
        args.output_text_file,
        args.output_tokens_file,
        args.image_path_file
    )

if __name__ == '__main__':
    main()