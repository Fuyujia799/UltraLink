import json
import sys

def count_non_english_records(jsonl_file_path):
    """
    Count the number of records in a JSONL file that do not contain the word "english" (case-insensitive).

    Parameters:
    - jsonl_file_path: The path to the JSONL file.

    Returns:
    - The count of records without the word "english".
    """
    non_english_count = 0

    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            record_str = json.dumps(record).lower()
            if "translate" not in record_str:
                non_english_count += 1

    return non_english_count

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py &lt;path_to_jsonl_file&gt;")
        sys.exit(1)

    jsonl_file_path = sys.argv[1]
    count = count_non_english_records(jsonl_file_path)
    print(f"Number of records without 'english': {count}")

if __name__ == "__main__":
    main()