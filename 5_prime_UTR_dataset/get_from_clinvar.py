import json
import os
import subprocess
import tempfile
import random
from multiprocessing import Pool, cpu_count

import pysam
import requests
import pandas as pd
from tqdm import tqdm

def _download_from_clinvar(tempdir):
    url = 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz'
    vcf_gz_path = os.path.join(tempdir, 'clinvar.vcf.gz')
    vcf_path = os.path.join(tempdir, 'clinvar.vcf')

    filters = {
        "5utr_benign.vcf": 'INFO/CLNSIG ~ "Benign" && INFO/MC ~ "SO:0001623"',
    }

    output_paths = [os.path.join(tempdir, fname) for fname in filters.keys()]
    if all(os.path.exists(p) for p in output_paths):
        return output_paths

    subprocess.run(f"wget {url} -O {vcf_gz_path}", shell=True, check=True)
    subprocess.run(f"gunzip -f {vcf_gz_path}", shell=True, check=True)

    files = []
    for output, expr in filters.items():
        out_path = os.path.join(tempdir, output)
        cmd = f"bcftools view -i '{expr}' {vcf_path} > {out_path}"
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        files.append(out_path)

    return files



def _parse_vcf_to_df(vcf_file):
    records = []
    with open(vcf_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            chrom, pos, _id, ref, alt, _qual, _filter, info = parts
            info_dict = dict(kv.split('=') for kv in info.split(';') if '=' in kv)
            record = {
                "chrom": chrom,
                "pos": int(pos),
                "ref": ref,
                "alt": alt,
                "gene": info_dict.get("GENEINFO", "").split(":")[0],
                "clnsig": info_dict.get("CLNSIG", ""),
                "mc": info_dict.get("MC", "")
            }
            records.append(record)
    return pd.DataFrame(records)


def _fetch_sequence_sampled(seq, chrom, pos, ref, extend, flank_min=200):
    if extend is None:
        extend = random.randint(600, 900)
    variant_rel_pos = random.randint(flank_min, extend - flank_min - len(ref))
    start = pos - variant_rel_pos
    end = start + extend - 1

    if start < 1:
        return None

    try:
        gt_seq = seq[start - 1:end]
    except Exception as e:
        print(f"[FASTA error] {chrom}:{start}-{end} – {e}")
        return None

    return gt_seq

def _process_sequences(args):
    chrom, group, fasta_path, extend = args
    if extend == 0:
        extend = None
    fasta_ref = pysam.FastaFile(fasta_path)
    results = []

    try:
        seq = fasta_ref.fetch(chrom)
    except:
        print(f"Failed to load {chrom} from FASTA")
        return []

    for _, row in group.iterrows():
        pos, ref, alt = row["pos"], row["ref"], row["alt"]
        wt_seq = _fetch_sequence_sampled(seq, chrom, pos, ref, extend)
        if wt_seq:
            results.append({
                "sequence": wt_seq,
                "label": 0,
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
            })
    return results

def get(dataset_location, filename, length=None):
    fasta_path = os.path.join(dataset_location, "Homo_sapiens.GRCh38.dna.primary_assembly.fa")
    files = _download_from_clinvar(dataset_location)

    df_all = pd.concat([_parse_vcf_to_df(f) for f in files], ignore_index=True)
    df_all.drop_duplicates(subset=["chrom", "pos", "ref", "alt"], inplace=True)

    num_workers = max(1, cpu_count() - 1)

    tasks = [
        (chrom, group, fasta_path, length)
        for chrom, group in df_all.groupby("chrom")
    ]

    print(f"Processing {len(tasks)} chromosomes across {cpu_count()} cores...")

    if length is None:
        length = 0

    with Pool(num_workers) as pool:
        results = pool.map(_process_sequences, tasks)

    dataset = [entry for result in results for entry in result]

    output_path = os.path.join(dataset_location, filename)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved dataset with {len(dataset)} entries to {output_path}")
    return dataset

if __name__ == "__main__":
    dataset_location = "/shared/5_utr/"
    length = 1200
    clinvar_filename = f"utr5_dataset_clinvar{f'_{length}' if length is not None else ''}.json"
    get(dataset_location, clinvar_filename, length)