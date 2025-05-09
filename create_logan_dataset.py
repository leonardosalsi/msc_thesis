import argparse
import os
import json
from datasets import Dataset, DatasetDict, Features, Value
from config import generated_datasets_dir, generator_cache_dir, logs_dir
import fasta_walker

def format_statistics(statistics):
    graph_length = statistics[0]
    num_nodes = statistics[1]
    num_sub_graphs = statistics[2]
    num_singletons = statistics[3]
    biggest_sub_graph = statistics[4]
    size_distribution = json.loads(statistics[5])

    return graph_length

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train model either from scratch or from pretrained weights with specified tokenization."
    )

    parser.add_argument(
        "fasta_files_path",
        type=str,
        help="Folder of FASTA files to be processed",
    )

    parser.add_argument(
        "--metadata_file_path",
        type=str,
        help="Path to metadata file",
    )

    parser.add_argument(
        "--acc_column",
        type=str,
        help="Column header name of accession in metadata file"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="Path where files are stored"
    )

    parser.add_argument(
        "--group_id_column",
        type=str,
        help="Column header name of group id of accession in metadata file"
    )

    parser.add_argument(
        "--kmer",
        type=int,
        default=31,
        help="Kmer length",
        choices=[31, 28, 25, 20]
    )

    parser.add_argument(
        "--reverse_complement",
        action="store_true",
        dest="reverse_complement",
        help="Also include reverse complements to graph."
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=0,
        help="Chunk size (defined when further splitting data)",
    )

    parser.add_argument(
        "--keep_remainder",
        action="store_true",
        dest="keep_remainder",
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Chunk size (defined when further splitting data)",
    )

    parser.add_argument(
        "--use_scratch",
        action="store_true",
        dest="use_scratch",
    )

    parser.add_argument(
        "--use_json",
        action="store_true",
        dest="use_json",
    )

    parser.add_argument(
        "--run_statistics",
        action="store_true",
        dest="run_statistics",
    )

    parser.add_argument(
        "--file_level",
        action="store_true",
        dest="file_level",
    )

    parser.add_argument(
        "--identity_threshold",
        type=float,
        default=0.0,
        help="Filter out sequences with identity value above this value",
    )

    return parser.parse_args()

if __name__ == "__main__":
        args = parse_args()
        max_workers = args.max_workers
        kmer = args.kmer
        chunk_size = args.chunk_size
        reverse_complement = args.reverse_complement
        fasta_files_path = args.fasta_files_path
        metadata_path = args.metadata_file_path
        output_path = args.output_path
        metadata_acc_column = args.acc_column
        metadata_group_id_column = args.group_id_column
        use_scratch = args.use_scratch
        use_json = args.use_json
        run_statistics = args.run_statistics
        file_level = args.file_level
        keep_remainder = args.keep_remainder
        identity_threshold = args.identity_threshold

        if run_statistics:
            gen = fasta_walker.run_statistics(
                kmer,
                reverse_complement,
                fasta_files_path,
                metadata_path,
                metadata_acc_column,
                metadata_group_id_column,
                max_workers,
                file_level
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
        if use_json:
            fasta_walker.create_random_walk_sequences_json(
                kmer,
                chunk_size,
                keep_remainder,
                reverse_complement,
                fasta_files_path,
                metadata_path,
                output_path,
                metadata_acc_column,
                metadata_group_id_column,
                max_workers,
                use_scratch,
                identity_threshold,
            )
        else:
            TMP_DIR = os.environ["TMPDIR"]
            print("TMPDIR:", TMP_DIR)
            if use_scratch:
                dataset_dir = os.path.join(TMP_DIR, 'datasets', f'logan')
                cache_dir = os.path.join(TMP_DIR, 'cache', 'logan')
            else:
                dataset_dir = os.path.join(generated_datasets_dir, f'logan')
                cache_dir = os.path.join(generator_cache_dir, 'logan')

            os.makedirs(cache_dir, exist_ok=True)
            if reverse_complement:
                dataset_dir = os.path.join(dataset_dir, f'kmer_{kmer}_reverse')
                generator_cache = os.path.join(cache_dir, f'kmer_{kmer}_reverse')
            else:
                dataset_dir = os.path.join(dataset_dir, f'kmer_{kmer}')
                generator_cache = os.path.join(cache_dir, f'kmer_{kmer}')
            dataset_dir = dataset_dir + f"_{chunk_size}k"
            cache_dir = cache_dir  + f"_{chunk_size}k"
            os.makedirs(dataset_dir, exist_ok=True)
            os.makedirs(cache_dir, exist_ok=True)

            features = Features({
                "sequence": Value("string"),
                f"{metadata_group_id_column}": Value("string"),
            })

            new_dataset = Dataset.from_generator(
                lambda: fasta_walker.create_random_walk_sequences(
                    kmer,
                    chunk_size,
                    keep_remainder,
                    reverse_complement,
                    fasta_files_path,
                    metadata_path,
                    output_path,
                    metadata_acc_column,
                    metadata_group_id_column,
                    max_workers,
                    use_scratch,
                    identity_threshold
                ),
                cache_dir=cache_dir,
                features=features
            )

            split_dataset = new_dataset.train_test_split(test_size=0.2, seed=112)
            train_dataset = split_dataset['train']
            test_dataset = split_dataset['test']
            dataset = DatasetDict({
                "train": train_dataset,
                "test": test_dataset
            })



            dataset.save_to_disk(dataset_dir, num_proc=max_workers)
            dataset_dir = dataset_dir + f"_filtered"
            cache_dir = cache_dir + f"_filtered"

            def filtered_generator(split):
                for example in split:
                    if len(example["sequence"]) == chunk_size:
                        yield example

            filtered_train = Dataset.from_generator(lambda: filtered_generator(dataset["train"]), cache_dir=cache_dir + "_train")
            filtered_test = Dataset.from_generator(lambda: filtered_generator(dataset["test"]), cache_dir=cache_dir + "_test")

            filtered_dataset = DatasetDict({
                "train": filtered_train,
                "test": filtered_test
            })

            filtered_dataset.save_to_disk(dataset_dir, num_proc=max_workers)
