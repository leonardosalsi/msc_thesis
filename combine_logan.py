from datasets import load_dataset, DatasetDict
import os

if __name__ == "__main__":

    json_dir = "/cluster/scratch/salsil/datasets/logan_1200"
    json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]
    dataset_full = load_dataset(
        "json",
        data_files=json_files,
        # jsonl=True,          # uncomment if each file is JSON-lines
        cache_dir="/cluster/scratch/salsil/.cache/upload",  # optional: speed up re-runs
    )
    dataset = dataset_full.shuffle(seed=101)
    validation_size = 500000
    dataset_train = dataset.select(range(validation_size, len(dataset)))
    dataset_validation = dataset.select(range(validation_size))

    ds =  DatasetDict({
        "train": dataset_train,
        "validation": dataset_validation
    })

    dataset.push_to_hub(
        repo_id=f"lsalsi/logan_multi_species_1k"
    )

    print(dataset)

