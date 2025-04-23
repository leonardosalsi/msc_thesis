import os
import subprocess
import tempfile
import random
import requests
import pandas as pd
from tqdm import tqdm

ENSEMBL_API = "https://rest.ensembl.org"


def download_from_clinvar(tempdir):
    url = 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz'
    vcf_gz_path = os.path.join(tempdir, 'clinvar.vcf.gz')
    vcf_path = os.path.join(tempdir, 'clinvar.vcf')
    subprocess.run(f"wget {url} -O {vcf_gz_path}", shell=True, check=True)
    subprocess.run(f"gunzip -f {vcf_gz_path}", shell=True, check=True)

    filters = {
        "5utr_pathogenic.vcf": 'INFO/CLNSIG ~ "Pathogenic" && INFO/MC ~ "SO:0001623"',
        "5utr_likely_pathogenic.vcf": 'INFO/CLNSIG ~ "Likely_pathogenic" && INFO/MC ~ "SO:0001623"',
        "5utr_likely_benign.vcf": 'INFO/CLNSIG ~ "Likely_benign" && INFO/MC ~ "SO:0001623"',
        "5utr_benign.vcf": 'INFO/CLNSIG ~ "Benign" && INFO/MC ~ "SO:0001623"',
    }

    files = []
    for output, expr in filters.items():
        out_path = os.path.join(tempdir, output)
        cmd = f"bcftools view -i '{expr}' {vcf_path} > {out_path}"
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        files.append(out_path)

    return files


def parse_vcf_to_df(vcf_file, label):
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
                "label": label,
                "clnsig": info_dict.get("CLNSIG", ""),
                "mc": info_dict.get("MC", "")
            }
            records.append(record)
    return pd.DataFrame(records)


def fetch_sequence_sampled(chrom, pos, ref, alt, seq_length=2200, flank_min=200):
    variant_rel_pos = random.randint(flank_min, seq_length - flank_min - len(ref))
    start = pos - variant_rel_pos
    end = start + seq_length - 1

    region = f"{chrom}:{max(1, start)}..{end}:1"
    url = f"{ENSEMBL_API}/sequence/region/human/{region}?"
    headers = {"Content-Type": "application/json"}
    r = requests.get(url, headers=headers)
    if not r.ok or "seq" not in r.json():
        print(f"Failed fetching {region}")
        return None, None

    wt_seq = r.json()["seq"].upper()
    if wt_seq[variant_rel_pos:variant_rel_pos+len(ref)] != ref:
        print(f"REF mismatch at {region} – expected {ref}, got {wt_seq[variant_rel_pos:variant_rel_pos+len(ref)]}")
        return None, None

    var_seq = wt_seq[:variant_rel_pos] + alt + wt_seq[variant_rel_pos+len(ref):]
    return wt_seq, var_seq, start, end


if __name__ == "__main__":
    sequence_length = 2200
    tempdir = os.path.join(tempfile.gettempdir(), "utr_dataset_gen")
    os.makedirs(tempdir, exist_ok=True)

    # Uncomment this to fetch and filter ClinVar data
    files = download_from_clinvar(tempdir)

    dataframes = []
    for file in files:
        label = 1 if "pathogenic" in file else 0
        dataframes.append(parse_vcf_to_df(file, label))

    df_all = pd.concat(dataframes, ignore_index=True)
    df_all.drop_duplicates(subset=["chrom", "pos", "ref", "alt"], inplace=True)

    sequences = []
    for _, row in tqdm(df_all.iterrows(), total=len(df_all)):
        wt_seq, var_seq, start, end = fetch_sequence_sampled(row['chrom'].replace("chr", ""), row['pos'], row['ref'], row['alt'], sequence_length)
        if wt_seq and var_seq:
            sequences.append({
                "chrom": row['chrom'],
                "pos": row['pos'],
                "ref": row['ref'],
                "alt": row['alt'],
                "label": row['label'],
                "wt_sequence": wt_seq,
                "variant_sequence": var_seq,
                "interval_start": start,
                "interval_end": end,
            })

    df_seqs = pd.DataFrame(sequences)
    out_seq_csv = os.path.join("/home/leonardo/Documents/", f"utr_variant_sequences_{sequence_length}.csv")
    df_seqs.to_csv(out_seq_csv, index=False)
    print(f"Saved sequence dataset: {out_seq_csv}")
