import os
import json
from argparse_dataclass import dataclass, ArgumentParser
from datasets import Dataset, DatasetDict, load_dataset

from config import logs_dir, datasets_cache_dir
import fasta_walker
from utils.util import print_args


def format_statistics(statistics):
    graph_length = statistics[0]
    num_nodes = statistics[1]
    num_sub_graphs = statistics[2]
    num_singletons = statistics[3]
    biggest_sub_graph = statistics[4]
    size_distribution = json.loads(statistics[5])

    return graph_length


@dataclass
class LoganCreateConfig:
    fasta_files_path: str = ""
    metadata_file_path: str = ""
    output_path: str = ""
    acc_column: str = ""
    group_id_column: str = ""
    kmer: int = 32
    max_workers: int = 1
    keep_remainder: bool = False
    use_scratch: bool = False
    reverse_complement: bool = False
    chunk_size: int = 1200
    run_statistics: bool = False
    file_level: bool = False
    identity_threshold: float = 0.85
    skip_json: bool = False


def parse_args():
    parser = ArgumentParser(LoganCreateConfig)
    return parser.parse_args()

if __name__ == "__main__":
        args = parse_args()
        _ = print_args(args, "LOGAN GENERATION ARGUMENTS")

        if args.run_statistics:
            gen = fasta_walker.run_statistics(
                args.kmer,
                args.reverse_complement,
                args.fasta_files_path,
                args.metadata_file_path,
                args.acc_column,
                args.group_id_column,
                args.max_workers,
                args.file_level
            )

            graph_length = 0
            num_nodes = 0
            num_sub_graphs = 0
            num_singletons = 0
            biggest_sub_graph = 0
            size_distribution = {}

            def accumulate_json(new_data):
                for key, value in new_data.items():
                    if key in size_distribution:
                        size_distribution[key] += value
                    else:
                        size_distribution[key] = value


            for g in gen:
                statistics = g['statistics']
                graph_length += statistics[0]
                num_nodes += statistics[1]
                num_sub_graphs += statistics[2]
                num_singletons += statistics[3]
                biggest_sub_graph += statistics[4]
                accumulate_json(json.loads(statistics[5]))

            final_stats = {
                "graph_length": graph_length,
                "num_nodes": num_nodes,
                "num_sub_graphs": num_sub_graphs,
                "num_singletons": num_singletons,
                "biggest_sub_graph": biggest_sub_graph,
                "size_distribution": size_distribution
            }
            output_file = os.path.join(logs_dir, "logan_stats.json")

            # Write the accumulated dictionary to the file
            with open(output_file, "w") as f:
                json.dump(final_stats, f, indent=4)
            exit(0)
        else:
            if not args.skip_json:
                fasta_walker.create_random_walk_sequences_json(
                    args.kmer,
                    args.chunk_size,
                    args.keep_remainder,
                    args.reverse_complement,
                    args.fasta_files_path,
                    args.metadata_file_path,
                    args.output_path,
                    args.acc_column,
                    args.group_id_column,
                    args.max_workers,
                    args.use_scratch,
                    args.identity_threshold,
                )
            folder_name = os.path.basename(args.output_path.rstrip('/'))
            dataset_dir = os.path.join(datasets_cache_dir, folder_name)
            os.makedirs(dataset_dir, exist_ok=True)

            file_list = [os.path.join(args.output_path, f) for f in os.listdir(args.output_path) if f.endswith('.json')]

            validation_size = 500000

            def gen():
                for file in file_list:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        for item in data:
                            yield {'sequence': item}

            dataset = Dataset.from_generator(gen)
            dataset = dataset.shuffle()
            train_dataset = dataset.select(range(validation_size, len(dataset)))
            validation_dataset = dataset.select(range(validation_size))

            dataset = DatasetDict({
                "train": train_dataset,
                "validation": validation_dataset
            })

            dataset.save_to_disk(dataset_dir)

            dataset_train = load_dataset(
                dataset_dir,
                cache_dir=datasets_cache_dir,
                split='train',
                trust_remote_code=True,
            )
            dataset_validation = load_dataset(
                dataset_dir,
                cache_dir=datasets_cache_dir,
                split='validation',
                trust_remote_code=True
            )

            print("TRAIN DATASET")
            print(dataset_train)
            print("VALIDATION DATASET")
            print(dataset_validation)




