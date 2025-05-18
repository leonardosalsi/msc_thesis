import json
import os
import subprocess
import random
from multiprocessing import Pool, cpu_count
from pprint import pprint

import pysam
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(base_dir, 'data')

def _download_from_clinvar(tempdir):
    url = 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz'
    vcf_gz_path = os.path.join(tempdir, 'clinvar.vcf.gz')
    vcf_path = os.path.join(tempdir, 'clinvar.vcf')

    filters = {
        "5utr_benign.vcf": 'INFO/CLNSIG ~ "Benign" && INFO/MC ~ "SO:0001623"',
        "5utr_pathogenic.vcf": 'INFO/CLNSIG ~ "Pathogenic" && INFO/MC ~ "SO:0001623"',
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
    label = 1 if 'pathogenic' in vcf_file else 0
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
                "mc": info_dict.get("MC", ""),
                "label": label,
            }
            records.append(record)
    return pd.DataFrame(records)


def _fetch_sequence_sampled(seq, chrom, pos, ref, alt, extend, flank_min=200):
    ref_len = len(ref)
    ref_ext = seq[pos - 1: pos - 1 + ref_len]
    assert ref_ext == ref, f"EXT Mismatch: got {ref_ext}, expected {ref}"

    mut_len = len(alt)
    mut = seq[:pos - 1] + alt + seq[pos:]

    if extend is None:
        extend = random.randint(600, 900)

    if mut_len > extend:
        return None

    success = False
    for i in range(10):
        growth = extend - mut_len + 1
        left_grow = random.randint(0, growth)
        right_grow = growth - left_grow
        gr_start = pos - left_grow
        gr_end = pos + right_grow
        if not (gr_start < 0 or gr_end >= len(seq)):
            success = True
            break
    if not success:
        return None

    start = gr_start
    end = gr_end
    mut = mut[start:end]

    return mut

def _process_sequences(args):
    chrom, group, fasta_path, extend, include_likely = args

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
        if 'Likely' in row['clnsig'] and not include_likely:
            continue
        mut_seq = _fetch_sequence_sampled(seq, chrom, pos, ref, alt, extend)
        if mut_seq:
            pprint({
                "sequence": mut_seq,
                "label": row['label'],
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
            })
            results.append({
                "sequence": mut_seq,
                "label": row['label'],
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
            })
    return results

def get_generator(length=None, include_likely=False):
    fasta_path = os.path.join(DATA_PATH, "Homo_sapiens.GRCh38.dna.primary_assembly.fa")
    files = _download_from_clinvar(DATA_PATH)
    df_all = pd.concat([_parse_vcf_to_df(f) for f in files], ignore_index=True)
    df_all.drop_duplicates(subset=["chrom", "pos", "ref", "alt"], inplace=True)

    MAX_YIELDS = 10000
    for i, (chrom, group) in enumerate(df_all.groupby("chrom")):
        yields = 0
        task = (chrom, group, fasta_path, length, include_likely)
        results = _process_sequences(task)

        for entry in results:
            yield entry
            yields += 1
            if yields >= MAX_YIELDS:
                break


if __name__ == "__main__":
    length = 2000
    include_likely = False
    gen = get_generator(length, include_likely)
    for g in gen:
        print(g)