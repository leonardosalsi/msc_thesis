import json
import os

SRC_DIR = '/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/generated_datasets/logan_raw_reserve'
DST_DIR = '/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/generated_datasets/logan_raw'

if __name__ == '__main__':
    files = ['1', '132', '28']
    for file in files:
        input_file = os.path.join(SRC_DIR, f'random_walk_{file}.json')
        output_file = os.path.join(DST_DIR, f'random_walk_{file}.json')
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Ensure the data is a list
        if not isinstance(data, list):
            raise ValueError("The JSON data must be an array of strings.")

        # Filter out strings that do not have exactly 1200 characters
        filtered_data = [s for s in data if isinstance(s, str) and len(s) == 1200]

        # Create the output directory if it does not exist
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the filtered list to the new JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=4)

        print("Filtered JSON data has been saved to:", output_file)