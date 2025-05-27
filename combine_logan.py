from datasets import load_dataset, DatasetDict
import os

if __name__ == "__main__":

    json_dir = "/cluster/scratch/salsil/datasets/logan_1200"
    json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]

    dataset_full = load_dataset(
        "json",
        data_files=json_files,
        block_size=1_000_000,  # ← add this!
        cache_dir="/cluster/scratch/salsil/.cache/upload",
    )

    # now split
    split = dataset_full["train"].train_test_split(
        test_size=500_000,  # or fraction: test_size=0.1
        seed=101
    )
    ds = DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })

    ds.push_to_hub(
        repo_id=f"lsalsi/logan_multi_species_1k"
    )


