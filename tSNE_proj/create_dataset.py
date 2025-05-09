import random
import os

from datasets import Dataset
from tqdm import tqdm
import pysam

from config import generated_datasets_dir
from tSNE_proj.generate_bed_files import get_files

FASTA_PATH = "/shared/DS/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
SEQUENCE_LENGTH = 6000
SAMPLES_PER_REGION = 10000

def parse_bed_file(file_path):
    with open(file_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                chrom = parts[0].removeprefix("chr")
                yield chrom, int(parts[1]), int(parts[2])

def sample_sequence(fasta, chrom, region_start, region_end, seq_len):
    region_len = region_end - region_start
    if region_len < 1:
        return None

    center = random.randint(region_start, region_end - 1)
    half_len = seq_len // 2
    seq_start = max(0, center - half_len)
    seq_end = seq_start + seq_len

    # Adjust if we're near the chromosome end (avoid short fetches)
    if fasta.get_reference_length(chrom) < seq_end:
        seq_start = fasta.get_reference_length(chrom) - seq_len
        seq_end = fasta.get_reference_length(chrom)

    if seq_start < 0:
        return None

    seq = fasta.fetch(chrom, seq_start, seq_end).upper()
    if len(seq) == seq_len and "N" not in seq:
        return seq, seq_start, seq_end
    return None

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    bed_files = get_files()

    def gen():
        fasta = pysam.FastaFile(FASTA_PATH)
        for label, bed_path in bed_files.items():
            print(f"Sampling from {label} regions...")
            sampled = 0
            for chrom, start, end in tqdm(parse_bed_file(bed_path), desc=f"{label}", unit="region"):
                result = sample_sequence(fasta, chrom, start, end, SEQUENCE_LENGTH)
                if result is None:
                    continue
                seq, seq_start, seq_end = result
                if seq:
                    label_id = list(bed_files.keys()).index(label)
                    yield {
                        "sequence": seq,
                        "label": label_id,
                        "region": label,
                        "chrom": chrom,
                        "start": seq_start,
                        "end": seq_end,
                        "region_start": start,
                        "region_end": end
                    }

                    sampled += 1
                if sampled >= SAMPLES_PER_REGION:
                    break
            print(f"Collected {sampled} sequences for {label}")

    dataset = Dataset.from_generator(gen)
    dataset.save_to_disk(os.path.join(generated_datasets_dir, f'tSNE_{SEQUENCE_LENGTH}'))
