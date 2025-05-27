import json

from datasets import load_dataset, DatasetDict, Dataset
import os

from config import cache_dir




if __name__ == "__main__":

    json_dir = "/cluster/scratch/salsil/datasets/logan_1200"
    json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]

    validation_size = 500000

    def gen():
        for file in json_files:
            with open(file, 'r') as f:
                data = json.load(f)
                for item in data:
                    yield {'sequence': item}

    generator_cache_dir = os.path.join(cache_dir, "generator")
    os.makedirs(generator_cache_dir, exist_ok=True)

    dataset = Dataset.from_generator(gen, cache_dir=generator_cache_dir)
    dataset = dataset.shuffle(seed=101)
    dataset_train = dataset.select(range(validation_size, len(dataset)))
    dataset_validation = dataset.select(range(validation_size))

    ds = DatasetDict({
        'train': dataset_train,
        'validation': dataset_validation,
    })

    ds.push_to_hub(
        repo_id=f"lsalsi/logan_multi_species_1k"
    )


