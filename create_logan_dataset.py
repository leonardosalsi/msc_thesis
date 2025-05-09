import os
import json
from argparse_dataclass import dataclass, ArgumentParser
from config import logs_dir
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
    fasta_files_path: str
    metadata_file_path: str
    output_path: str
    acc_column: str
    group_id_column: str
    kmer: int = 32
    max_workers: int = 1
    keep_remainder: bool = False
    use_scratch: bool = False
    reverse_complement: bool = False
    chunk_size: int = 1200
    run_statistics: bool = False
    file_level: bool = False
    identity_threshold: float = 0.85


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
