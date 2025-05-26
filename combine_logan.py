import os
import json

from tqdm import tqdm


def merge_json_arrays(input_dir, output_file):
    merged = []

    # Iterate over every file in the folder
    for fname in tqdm(os.listdir(input_dir)):
        if fname.lower().endswith('.json'):
            full_path = os.path.join(input_dir, fname)
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    merged.extend(data)
                else:
                    raise ValueError(f"{fname} does not contain a JSON array")

    # Write out the combined array
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(merged, f_out, indent=2)

if __name__ == "__main__":
    folder_path = "/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/generated_datasets/logan/logan_1200"
    output_path = "/cluster/work/grlab/projects/projects2024-petagraph-input-optimisation-msc-thesis/generated_datasets/logan/logan_1200.json"
    merge_json_arrays(folder_path, output_path)
    print(f"Merged {folder_path} → {output_path}")
