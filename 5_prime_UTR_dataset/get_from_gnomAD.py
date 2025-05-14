import json
import os
import random
import pandas as pd
from gnomad_db.database import gnomAD_DB
import pysam
from multiprocessing import Pool, cpu_count

base_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(base_dir, 'data')

def _load_five_prime_utrs_from_gtf(gtf_path):
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

def _get_utr_dataframe(gtf_path, cache_path) -> pd.DataFrame:
    if os.path.exists(cache_path):
        print(f"Loading 5′UTR DataFrame from cache: {cache_path}")
        return pd.read_parquet(cache_path)
    else:
        print(f"Extracting 5′UTR regions from GTF: {gtf_path}")
        utr_df = _load_five_prime_utrs_from_gtf(gtf_path)
        utr_df.to_parquet(cache_path, index=False)
        print(f"Saved 5′UTR DataFrame to: {cache_path}")
        return utr_df

def _mutate_sequence(seq, start, pos, ref, alt):
    local_pos = pos - start
    return seq[:local_pos] + alt + seq[local_pos + len(ref):]

def _make_sequence_exact_length(extend, start, pos, end, mut_seq, seq):
    mut_seq_len = len(mut_seq)
    max_len = len(seq)
    if mut_seq_len < extend:
        right_start = end + 1
        to_add = extend - mut_seq_len
        left_len = random.randint(0, to_add)
        right_len = to_add - left_len

        left_start = start - left_len - 1
        left_end = start - 1
        right_end = start + mut_seq_len - 1 + right_len - 1

        if right_end > max_len:
            overflow = right_end - max_len
            right_len -= overflow
            left_len += overflow
            right_end = max_len
            left_start = start - left_len - 1
            left_end = start - 1

        if left_start < 1:
            underflow = 1 - left_start
            left_len -= underflow
            right_len += underflow
            left_start = 1
            left_end = start - 1
            right_end = start + mut_seq_len - 1 + right_len - 1

        left_seq = seq[left_start-1:left_end]
        right_seq = seq[right_start-1:right_end]
        mut_seq = left_seq + mut_seq + right_seq
        seq_start = left_start

    elif mut_seq_len > extend:
        diff = mut_seq_len - extend
        variant_offset = pos - start
        trim_start = min(variant_offset, diff)
        mut_seq =  mut_seq[trim_start:trim_start + extend]
        seq_start = start + trim_start

    else:
        seq_start = start

    return mut_seq, seq_start

def _get_label(AF):
    if AF >= 0.05:
        return 0
    elif AF <= 0.0001:
        return 1
    else:
        return None

def _process_af_sequences(args):
    if len(args) == 6:
        chrom, utr_group, fasta_path, db_path, extend, return_af = args
    else:
        chrom, utr_group, fasta_path, db_path, extend = args

    if extend == 0:
        extend = None
    fasta_ref = pysam.FastaFile(fasta_path)
    db = gnomAD_DB(db_path, gnomad_version="v4")
    results = []

    seq = fasta_ref.fetch(chrom).upper()
    seq_len = len(seq)

    for _, region_info in utr_group.iterrows():
        start, end, strand = region_info["start"], region_info["end"], region_info["strand"]
        reg_len = end - start + 1
        if reg_len > extend:
            continue

        success = False
        for i in range(10):
            growth = extend - reg_len + 1
            left_grow = random.randint(0, growth)
            right_grow = growth - left_grow
            gr_start = start - left_grow
            gr_end = end + right_grow
            if not (gr_start < 0 or gr_end >= seq_len):
                success = True
                break
        if not success:
            continue

        start = gr_start
        end = gr_end
        gt_seq = seq[start - 1:end]
        variant_info_db = db.get_info_for_interval(chrom=chrom, interval_start=start, interval_end=end,
                                                   query="chrom,pos,ref,alt,AF")
        for _, var in variant_info_db.iterrows():
            AF = var["AF"]
            if AF is None or len(var["alt"]) != 1 or len(var["ref"]) != 1:
                continue
            pos, ref, alt = var["pos"], var["ref"], var["alt"]
            seq_start = start
            rel_pos = pos - seq_start
            assert gt_seq[rel_pos] == ref, f"Mismatch: got {gt_seq[rel_pos]}, expected {ref}"
            label = _get_label(AF)
            if label is None:
                continue
            results.append({
                'sequence': gt_seq,
                'label': label,
                'chrom': chrom,
                'pos': rel_pos,
                'ref': ref,
                'alt': alt,
                'af': AF,
                'start': seq_start,
            })

    return results

def _process_sequences(args):
    if len(args) == 6:
        chrom, utr_group, fasta_path, db_path, extend, return_af = args
    else:
        chrom, utr_group, fasta_path, db_path, extend = args
        return_af = False
    if extend == 0:
        extend = None
    fasta_ref = pysam.FastaFile(fasta_path)
    db = gnomAD_DB(db_path, gnomad_version="v4")
    results = []

    try:
        seq = fasta_ref.fetch(chrom).upper()
    except:
        return []

    for _, region_info in utr_group.iterrows():
        start, end, strand = region_info["start"], region_info["end"], region_info["strand"]
        gt_seq = seq[start - 1:end]

        variant_info_db = db.get_info_for_interval(chrom=chrom, interval_start=start, interval_end=end,
                                                   query="chrom,pos,ref,alt,AF")
        for _, var in variant_info_db.iterrows():
            AF = var["AF"]
            if AF is None or len(var["alt"]) != 1 or len(var["ref"]) != 1:
                continue
            pos, ref, alt = var["pos"], var["ref"], var["alt"]
            mut = _mutate_sequence(gt_seq, start, pos, ref, alt)
            if mut is None:
                continue
            seq_start = start
            if extend:
                mut, seq_start = _make_sequence_exact_length(extend, start, pos, end, mut, seq)
            if mut is None:
                continue
            label = _get_label(AF)
            if label is None:
                continue
            if extend and len(mut) != extend:
                continue
            if return_af:
                results.append({
                    'sequence': mut,
                    'label': label,
                    'chrom': chrom,
                    'pos': pos,
                    'ref': ref,
                    'alt': alt,
                    'af': AF,
                    'start': seq_start,
                })
            else:
                results.append({
                    'sequence': mut,
                    'label': label,
                    'chrom': chrom,
                    'pos': pos,
                    'ref': ref,
                    'alt': alt,
                })
    return results

def get(filename, length=None, return_af=False):
    gtf_file = os.path.join(DATA_PATH, "Homo_sapiens.GRCh38.110.gtf")
    utr_cache_file = os.path.join(DATA_PATH, "utr5_dataframe.parquet")
    fasta_path = os.path.join(DATA_PATH, "Homo_sapiens.GRCh38.dna.primary_assembly.fa")

    num_workers = max(1, cpu_count() - 1)
    utr5_df = _get_utr_dataframe(gtf_file, utr_cache_file)

    if length is None:
        length = 0

    tasks = [
        (chrom, group, fasta_path, DATA_PATH, length, return_af)
        for chrom, group in utr5_df.groupby("chrom")
    ]

    print(f"Processing {len(tasks)} chromosomes across {num_workers} workers")

    with Pool(num_workers) as pool:
        results = pool.map(_process_sequences, tasks)

    dataset = [entry for result in results for entry in result]

    output_path = os.path.join(DATA_PATH, filename)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved dataset with {len(dataset)} entries to {output_path}")
    return dataset

def get_generator(length=None, return_af=False):
    gtf_file = os.path.join(DATA_PATH, "Homo_sapiens.GRCh38.110.gtf")
    utr_cache_file = os.path.join(DATA_PATH, "utr5_dataframe.parquet")
    fasta_path = os.path.join(DATA_PATH, "Homo_sapiens.GRCh38.dna.primary_assembly.fa")

    utr5_df = _get_utr_dataframe(gtf_file, utr_cache_file)
    MAX_YIELDS = 10000
    if length is None:
        length = 0

    for i, (chrom, group) in enumerate(utr5_df.groupby("chrom")):
        yields = 0
        task = (chrom, group, fasta_path, DATA_PATH, length, return_af)
        results = _process_af_sequences(task)

        for entry in results:
            yield entry
            yields += 1
            if yields >= MAX_YIELDS:
                break

        print(f"Chrom {chrom} done. Yields: {yields}")


if __name__ == "__main__":
    length = 6000
    gen = get_generator(length, True)
    for g in gen:
        print(g)

