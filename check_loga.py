import os
import json
import glob

def sum_json_array_lengths(directory):
    total_entries = 0
    # Use glob to find all JSON files in the directory.
    json_files = glob.glob(os.path.join(directory, "*.json"))
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    total_entries += len(data)
                else:
                    print(f"Warning: {file_path} does not contain a JSON array.")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return total_entries

if __name__ == "__main__":
    # Specify the directory containing your JSON files.
    directory = "/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/generated_datasets/logan_1k"
    total = sum_json_array_lengths(directory)
    print("Total entries across all files:", total)
