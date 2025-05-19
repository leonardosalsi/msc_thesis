import random
import os
from datasets import Dataset
from tqdm import tqdm
import pysam
from config import generated_datasets_dir
from genomic_elements_dataset.generate_bed_files import get_files
from utils.util import gc_content

base_dir = os.path.dirname(os.path.abspath(__file__))
FASTA_PATH = os.path.join(base_dir, "data/Homo_sapiens.GRCh38.dna.primary_assembly.fa")
EXTEND_TO_NORM = True
SEQUENCE_LENGTH = 6000
SAMPLES_PER_REGION = 10000
MAX_SEQUENCE_LENGTH = 6000

def parse_bed_file(file_path):
    with open(file_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                chrom = parts[0].removeprefix("chr")
                yield chrom, int(parts[1]), int(parts[2])

def get_unique_chromosomes_from_bed(file_path, strip_chr_prefix=True):
    chroms = set()
    with open(file_path) as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue  # skip comments or empty lines
            parts = line.strip().split()
            if len(parts) >= 3:
                chrom = parts[0]
                if strip_chr_prefix:
                    chrom = chrom.removeprefix("chr")
                chroms.add(chrom)
    return sorted(chroms)

def sample_sequence(fasta, chrom, region_start, region_end, chrom_len):
    if not EXTEND_TO_NORM:
        seq = fasta.fetch(chrom, region_start, region_end).upper()
        if "N" not in seq:
            return seq, region_start, region_end
        return None

    reg_len = region_end - region_start + 1
    if reg_len > SEQUENCE_LENGTH:
        max_start_offset = reg_len - SEQUENCE_LENGTH
        offset = random.randint(0, max_start_offset)
        start = region_start + offset
        end = start + SEQUENCE_LENGTH

        local_region_start = 0
        local_region_end = SEQUENCE_LENGTH
    else:
        start = region_start
        end = region_end
        success = False
        for i in range(10):
            growth = SEQUENCE_LENGTH - reg_len + 1
            left_grow = random.randint(0, growth)
            right_grow = growth - left_grow
            start = region_start - left_grow
            end = region_end + right_grow
            if not (start < 0 or end >= chrom_len):
                success = True

                break

        if not success:
            print("UNABLE TO EXTEND")
            return None

        local_region_start = region_start - start
        local_region_end = region_end - start

    seq = fasta.fetch(chrom, start, end).upper()

    """
    Extract critical region like so
    seq[local_region_start:local_region_end]
    """

    if len(seq) == SEQUENCE_LENGTH and "N" not in seq:
        return seq, local_region_start, local_region_end
    return None

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    bed_files = get_files()

    def gen():
        fasta = pysam.FastaFile(FASTA_PATH)
        for label, bed_path in bed_files.items():
            print(f"Sampling from {label} regions...")
            sampled = 0
            chroms = get_unique_chromosomes_from_bed(bed_path)
            chrom_len = {}
            for chrom in tqdm(chroms, desc=f"{label}", unit="chromosome"):
                try:
                    chrom_len[chrom] = fasta.get_reference_length(chrom)
                except:
                    continue

            for chrom, start, end in tqdm(parse_bed_file(bed_path), desc=f"{label}", unit="region"):
                result = sample_sequence(fasta, chrom, start, end, chrom_len[chrom])
                if result is None:
                    continue
                seq, loc_start, loc_end = result

                if seq is None:
                    continue

                region = seq[loc_start:loc_end]
                label_id = list(bed_files.keys()).index(label)

                yield {
                    "sequence": seq,
                    "label": label_id,
                    "region": label,
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "loc_start": loc_start,
                    "loc_end": loc_end,
                    "seq_gc": gc_content(seq),
                    "region_gc": gc_content(region)
                }
                sampled += 1

                if sampled >= SAMPLES_PER_REGION:
                    break

            print(f"Collected {sampled} sequences for {label}")

    dataset = Dataset.from_generator(gen)
    dataset.info.dataset_name = f'genomic_elements'
    dataset.save_to_disk(os.path.join(generated_datasets_dir, f'genomic_elements'))
