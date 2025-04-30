import os
from pprint import pprint
from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
from gnomad_db.database import gnomAD_DB

# Set paths
dataset_location = "/shared/5_utr/"
database_location = dataset_location
gtf_file = os.path.join(database_location, "Homo_sapiens.GRCh38.110.gtf")
utr_cache_file = os.path.join(dataset_location, "utr5_dataframe.parquet")
fasta_path = os.path.join(dataset_location, "Homo_sapiens.GRCh38.dna.primary_assembly.fa")

# Initialize DB
db = gnomAD_DB(database_location, gnomad_version="v4")


def load_five_prime_utrs_from_gtf(gtf_path: str) -> pd.DataFrame:
    col_names = ["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]

    gtf = pd.read_csv(
        gtf_path,
        sep="\t",
        comment="#",
        names=col_names,
        dtype={"seqname": str},
        low_memory=False
    )

    utr5 = gtf[gtf["feature"] == "five_prime_utr"].copy()

    def extract_attr(attr_str, key):
        for part in attr_str.split(';'):
            part = part.strip()
            if part.startswith(f"{key} "):
                return part.split('"')[1]
        return None

    utr5["gene_id"] = utr5["attribute"].apply(lambda a: extract_attr(a, "gene_id"))
    utr5["transcript_id"] = utr5["attribute"].apply(lambda a: extract_attr(a, "transcript_id"))

    utr5 = utr5.rename(columns={"seqname": "chrom"})
    utr5 = utr5[["chrom", "start", "end", "strand", "gene_id", "transcript_id"]]

    return utr5

def get_utr_dataframe(gtf_path: str, cache_path: str) -> pd.DataFrame:
    if os.path.exists(cache_path):
        print(f"Loading 5′UTR DataFrame from cache: {cache_path}")
        return pd.read_parquet(cache_path)
    else:
        print(f"Extracting 5′UTR regions from GTF: {gtf_path}")
        utr_df = load_five_prime_utrs_from_gtf(gtf_path)
        utr_df.to_parquet(cache_path, index=False)
        print(f"Saved 5′UTR DataFrame to: {cache_path}")
        return utr_df

def create_fasta_index(fasta_path: str):
    """
    Loads the entire genome into a dictionary of SeqRecord objects.
    """
    print(f"Indexing FASTA: {fasta_path}")
    return SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))

def fetch_reference_sequence(records, chrom: str, start: int, end: int, strand: str = "+") -> str:
    """
    Fetch sequence from preloaded genome dict.
    """
    chrom = chrom.lstrip("chr")
    if chrom not in records:
        print(f"Chromosome {chrom} not found.")
        return None

    seq = records[chrom].seq[start - 1:end]  # 0-based
    return str(seq.upper()) if strand == "+" else str(seq.reverse_complement().upper())

def mutate_sequence(wt_seq: str, utr_start: int, variant_pos: int, ref: str, alt: str) -> str | None:
    rel_pos = variant_pos - utr_start
    if rel_pos < 0 or rel_pos + len(ref) > len(wt_seq):
        print(f"Variant out of range: rel_pos={rel_pos}, ref_len={len(ref)}, seq_len={len(wt_seq)}")
        return None

    if wt_seq[rel_pos:rel_pos + len(ref)] != ref:
        print(f"Reference mismatch at position {variant_pos}: expected {ref}, got {wt_seq[rel_pos:rel_pos + len(ref)]}")
        return None

    return wt_seq[:rel_pos] + alt + wt_seq[rel_pos + len(ref):]


if __name__ == "__main__":
    utr5_df = get_utr_dataframe(gtf_file, utr_cache_file)
    genome_records = create_fasta_index(fasta_path)
    i = 0
    for _, region_info in utr5_df.iterrows():
        i += 1
        chrom, start, end, strand = region_info["chrom"], region_info["start"], region_info["end"], region_info["strand"]

        gt_seq = fetch_reference_sequence(genome_records, chrom, start, end, strand)

        variant_info_db = db.get_info_for_interval(chrom=chrom, interval_start=start, interval_end=end, query="chrom,pos,ref,alt,AF")
        for _, variant_info in variant_info_db.iterrows():
            AF = variant_info['AF']
            pos = variant_info['pos']
            ref = variant_info['ref']
            alt = variant_info['alt']
            mut = mutate_sequence(gt_seq, start, pos, ref, alt)
            print(pos, ref, alt)
            print(gt_seq)
            print(mut)
            print()

        if i == 100:
            break  # just do one for demo

