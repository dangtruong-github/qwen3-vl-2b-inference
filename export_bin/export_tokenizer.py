import json
import struct
import os
import argparse

def create_tokenizer_bin(tokenizer_json_path: str, tokenizer_config_path: str, output_bin_path: str):
    """
    Creates a binary tokenizer file from tokenizer.json and tokenizer_config.json.

    Args:
        tokenizer_json_path: Path to the Hugging Face 'tokenizer.json' file.
        tokenizer_config_path: Path to the Hugging Face 'tokenizer_config.json' file.
        output_bin_path: Path to the output binary file 'tokenizer.bin'.
    """
    # Load the tokenizer configuration files
    try:
        with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: tokenizer.json not found at {tokenizer_json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {tokenizer_json_path}")
        return

    try:
        with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
            tokenizer_config = json.load(f)
    except FileNotFoundError:
        print(f"Error: tokenizer_config.json not found at {tokenizer_config_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {tokenizer_config_path}")
        return

    # Extract vocabulary from tokenizer.json
    vocab = tokenizer_data['model']['vocab']

    # Extract merges from tokenizer.json
    merges = tokenizer_data['model']['merges']

    # Get added tokens from tokenizer_config
    added_tokens = tokenizer_config.get('added_tokens_decoder', {}) # Use .get with default empty dict for robustness

    # Combine base vocabulary with added tokens
    full_vocab = {}

    # Add base vocabulary
    for token, token_id in vocab.items():
        full_vocab[token_id] = token

    # Add special tokens from added_tokens_decoder
    for token_id_str, token_info in added_tokens.items():
        try:
            token_id = int(token_id_str)
            token_content = token_info['content']
            full_vocab[token_id] = token_content
        except ValueError:
            print(f"Warning: Skipping non-integer token ID '{token_id_str}' in added_tokens_decoder.")
        except KeyError:
            print(f"Warning: Skipping token ID '{token_id_str}' missing 'content' in added_tokens_decoder.")

    # Create the vocabulary array in order
    vocab_size = len(full_vocab)
    # Check for contiguous token IDs and fill the list
    if vocab_size > 0:
        max_id = max(full_vocab.keys())
    else:
        max_id = -1
        
    vocab_list = [None] * (max_id + 1) # Initialize list to cover all IDs up to max_id
    
    for token_id, token in full_vocab.items():
        if token_id < len(vocab_list):
            vocab_list[token_id] = token
            
    # Remove any None entries if the IDs are not contiguous or if max_id was higher than expected vocab_size
    # We should only write the tokens that were actually in the vocab/added_tokens
    final_vocab_list = [token for token in vocab_list if token is not None]
    vocab_size = len(final_vocab_list) # Recalculate size based on non-None tokens if necessary, though ideally it should be len(full_vocab) if IDs are contiguous starting from 0.
    # The previous logic should ensure vocab_size is correct based on full_vocab, but let's stick to the list created from IDs.

    # Recalculate vocab_size based on full_vocab for consistency
    vocab_size = len(full_vocab)
    # Ensure vocab_list is in order 0 to vocab_size-1
    vocab_list_ordered = [full_vocab[i] for i in range(vocab_size) if i in full_vocab]
    
    if len(vocab_list_ordered) != vocab_size:
        print("Error: Token IDs are not contiguous starting from 0. Cannot create ordered list.")
        # Fallback to the original logic which was intended to handle non-contiguous but might be buggy
        # For simplicity and typical tokenizer structure, we assume contiguous IDs from 0 to vocab_size-1
        # Reverting to simple construction based on sorted keys:
        ordered_keys = sorted(full_vocab.keys())
        if ordered_keys != list(range(vocab_size)):
             print(f"Error: Token IDs are not exactly contiguous from 0 to {vocab_size-1}.")
             # Proceeding with the sorted list, but the binary file reader might expect 0..N-1 contiguous IDs
        
        vocab_list_ordered = [full_vocab[k] for k in ordered_keys]
        vocab_size = len(vocab_list_ordered) # Update vocab size based on actual tokens found

    # Calculate max token length
    if not vocab_list_ordered:
        max_token_length = 0
    else:
        max_token_length = max(len(token.encode('utf-8')) for token in vocab_list_ordered)


    # Write to binary file
    try:
        with open(output_bin_path, 'wb') as f:
            # Write vocab_size (4-byte integer 'i')
            f.write(struct.pack('I', vocab_size))

            # Write merges_size (4-byte integer 'i')
            f.write(struct.pack('I', len(merges)))

            # Write max_token_length (4-byte integer 'i')
            f.write(struct.pack('I', max_token_length))

            # Write vocabulary
            for token in vocab_list_ordered:
                token_bytes = token.encode('utf-8')
                # Write length of token string (4-byte integer 'i')
                f.write(struct.pack('I', len(token_bytes)))
                # Write token string
                f.write(token_bytes)

            # Write merges
            for merge in merges:
                merge_bytes = merge.encode('utf-8')
                # Write length of merge string (4-byte integer 'i')
                f.write(struct.pack('I', len(merge_bytes)))
                # Write merge string
                f.write(merge_bytes)

    except IOError as e:
        print(f"Error writing to output file {output_bin_path}: {e}")
        return

    print(f"Successfully created binary tokenizer file: {output_bin_path}")
    print(f"  - Vocabulary size (including added tokens): {vocab_size}")
    print(f"  - Merges count: {len(merges)}")
    print(f"  - Max token length (bytes): {max_token_length}")
    print(f"  - Added tokens count: {len(added_tokens)}")


def main():
    parser = argparse.ArgumentParser(description="Convert Hugging Face tokenizer files (tokenizer.json and tokenizer_config.json) into a custom binary format.")
    
    # Add arguments
    parser.add_argument(
        '--tokenizer_json',
        type=str,
        default='tokenizer.json',
        help="Path to the Hugging Face 'tokenizer.json' file (default: tokenizer.json)."
    )
    parser.add_argument(
        '--tokenizer_config',
        type=str,
        default='tokenizer_config.json',
        help="Path to the Hugging Face 'tokenizer_config.json' file (default: tokenizer_config.json)."
    )
    parser.add_argument(
        '--output_bin',
        type=str,
        default='tokenizer.bin',
        help="Path for the output binary file (default: tokenizer.bin)."
    )

    # Parse arguments
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    create_tokenizer_bin(args.tokenizer_json, args.tokenizer_config, args.output_bin)

if __name__ == "__main__":
    main()