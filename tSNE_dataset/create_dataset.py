import random
import os

from datasets import Dataset
from tqdm import tqdm
import pysam

from config import generated_datasets_dir
from tSNE_dataset.generate_bed_files import get_files


base_dir = os.path.dirname(os.path.abspath(__file__))
FASTA_PATH = os.path.join(base_dir, "data/Homo_sapiens.GRCh38.dna.primary_assembly.fa")
EXTEND_TO_NORM = False
SEQUENCE_LENGTH = 6000
SAMPLES_PER_REGION = 10000

def get_cg_content(seq):
    """
    Calculation via Brock Biology of Microorganisms 10th edition
    @book{Madigan_Martinko_Parker_2003, place={Upper Saddle River, NJ}, title={Brock Biology of Microorganisms}, publisher={Prentice Hall/Pearson Education}, author={Madigan, Michael T. and Martinko, John M. and Parker, Jack}, year={2003}}
    """
    full_len = len(seq)
    num_GC = seq.count("G") + seq.count("C")
    return num_GC / full_len * 100

def parse_bed_file(file_path):
    with open(file_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                chrom = parts[0].removeprefix("chr")
                yield chrom, int(parts[1]), int(parts[2])

def sample_sequence(fasta, chrom, region_start, region_end, seq_len):
    if not EXTEND_TO_NORM:
        seq = fasta.fetch(chrom, region_start, region_end).upper()
        if "N" not in seq:
            return seq, region_start, region_end
        return None

    region_len = region_end - region_start
    if region_len < 1 or region_len > seq_len:
        return None  # Skip invalid or too-large regions

    # Compute the bounds for valid seq_start
    min_seq_start = max(0, region_end - seq_len)
    max_seq_start = region_start

    if min_seq_start > max_seq_start:
        return None  # Not enough space to include the full region

    seq_start = random.randint(min_seq_start, max_seq_start)
    seq_end = seq_start + seq_len

    if fasta.get_reference_length(chrom) < seq_end:
        return None  # Window goes beyond chromosome end

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

                region_relative_start = start - seq_start
                region_relative_end = end - seq_start
                region = seq[region_relative_start:region_relative_end]

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
                        "region_end": end,
                        "seq_gc": get_cg_content(seq),
                        "region_gc": get_cg_content(region)
                    }

                    sampled += 1
                if sampled >= SAMPLES_PER_REGION:
                    break
            print(f"Collected {sampled} sequences for {label}")

    dataset = Dataset.from_generator(gen)
    dataset.save_to_disk(os.path.join(generated_datasets_dir, f'tSNE_{SEQUENCE_LENGTH}{"_var" if not EXTEND_TO_NORM else ""}'))
