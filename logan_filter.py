import os
import json


def filter_json_files(source_dir, dest_dir, min_length=1200):
    # Ensure destination directory exists.
    os.makedirs(dest_dir, exist_ok=True)

    # Iterate over all files in the source directory.
    for filename in os.listdir(source_dir):
        if filename.endswith('.json'):
            source_file = os.path.join(source_dir, filename)
            dest_file = os.path.join(dest_dir, filename)

            # Open and load the JSON file.
            with open(source_file, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in {source_file}: {e}")
                    continue

            # Check if data is a list.
            if not isinstance(data, list):
                print(f"Skipping {source_file}: JSON data is not an array.")
                continue

            # Filter out strings shorter than min_length.
            filtered_data = [s for s in data if isinstance(s, str) and len(s) >= min_length]

            # Write the filtered data to the destination file.
            with open(dest_file, 'w') as f:
                json.dump(filtered_data, f, indent=2)
            print(f"Processed {filename}: {len(filtered_data)} entries saved.")


if __name__ == '__main__':
    source_directory = "/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/generated_datasets/logan_1"
    destination_directory = "/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/generated_datasets/logan_1k"

    filter_json_files(source_directory, destination_directory)
