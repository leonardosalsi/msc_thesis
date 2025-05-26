import os
from collections import defaultdict
import gffutils

"""
1. GENCODE GTF annotation:
    - Source: https://www.gencodegenes.org/human/release_38.html
    - Example: gencode.v38.annotation.gtf

2. Genome FASTA and index:
    - wget https://ftp.ensembl.org/pub/release-114/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
    - gunzip Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
    - samtools faidx Homo_sapiens.GRCh38.dna.primary_assembly.fa

This script generates BED files for different genomic region types:
    - Coding regions (CDS)
    - 5′ UTR and 3′ UTR (split from general UTRs)
    - Introns (inferred between exons of each transcript)
    - Intergenic regions (regions not covered by any transcript)

These BED files are used to sample labeled sequences from the genome for downstream analyses.
"""

base_dir = os.path.dirname(os.path.abspath(__file__))
GENCODE_FILE = os.path.join(base_dir, 'data/gencode.v38.annotation.gtf')
GENCODE_DB = os.path.join(base_dir, 'data/gencode.v38.db')
GENOME_FAI= os.path.join(base_dir, 'data/Homo_sapiens.GRCh38.dna.primary_assembly.fa.fai')


def get_gencode_db():
    """
    Loads or creates a gffutils database from the GENCODE GTF file.
    This allows efficient querying of gene features (like CDS, exon, UTR).
    """
    if not os.path.exists(GENCODE_DB):
        gffutils.create_db(
            GENCODE_FILE,
            dbfn=GENCODE_DB,
            force=True,
            keep_order=True,
            disable_infer_genes=True,
            disable_infer_transcripts=True
        )
    return gffutils.FeatureDB(GENCODE_DB)

def create_cds_bed_file(db):
    """
    Extracts all CDS (coding sequence) regions from the GENCODE database.
    These are written to cds.bed in 0-based BED format.
    """
    output_file = os.path.join(base_dir, 'bed_files/cds.bed')
    if not os.path.exists(output_file):
        with open(output_file, "w") as out:
            for feature in db.features_of_type("CDS"):
                chrom = feature.chrom
                start = feature.start - 1
                end = feature.end
                strand = feature.strand
                out.write(f"{chrom}\t{start}\t{end}\t.\t0\t{strand}\n")
    return output_file

def create_five_three_prime_bed_files(db):
    """
    Separates UTR features into 5′ and 3′ UTRs based on their position
    relative to CDS regions within the same transcript.

    - On the '+' strand (upstream):
        - UTRs before CDS = 5′ UTR
        - UTRs after CDS  = 3′ UTR
    - On the '−' strand (downstream):
        - UTRs after CDS  = 5′ UTR
        - UTRs before CDS = 3′ UTR
    """
    output_file_five = os.path.join(base_dir, 'bed_files/five_prime_utr.bed')
    output_file_three = os.path.join(base_dir, 'bed_files/three_prime_utr.bed')
    if not os.path.exists(output_file_five) or not os.path.exists(output_file_three):
        utr_by_tx = defaultdict(list)
        cds_by_tx = defaultdict(list)

        for utr in db.features_of_type("UTR"):
            tx_id = utr.attributes.get("transcript_id", [None])[0]
            if tx_id:
                utr_by_tx[tx_id].append(utr)

        for cds in db.features_of_type("CDS"):
            tx_id = cds.attributes.get("transcript_id", [None])[0]
            if tx_id:
                cds_by_tx[tx_id].append(cds)

        with open(output_file_five, "w") as out5, open(output_file_three, "w") as out3:
            for tx_id, utrs in utr_by_tx.items():
                if tx_id not in cds_by_tx:
                    continue

                cds_list = cds_by_tx[tx_id]
                cds_start = min(c.start for c in cds_list)
                cds_end = max(c.end for c in cds_list)
                strand = cds_list[0].strand
                chrom = cds_list[0].chrom

                for utr in utrs:
                    start = utr.start - 1
                    end = utr.end
                    out = None

                    if strand == "+":
                        if utr.end <= cds_start:
                            out = out5
                        elif utr.start >= cds_end:
                            out = out3
                    elif strand == "-":
                        if utr.start >= cds_end:
                            out = out5
                        elif utr.end <= cds_start:
                            out = out3

                    if out:
                        out.write(f"{chrom}\t{start}\t{end}\t.\t0\t{strand}\n")
    return output_file_five, output_file_three

def create_intron_bed_file(db):
    """
    Infers introns from exon boundaries within each transcript.
    For transcripts with ≥2 exons, introns are the gaps between adjacent exons.
    """
    output_file = os.path.join(base_dir, 'bed_files/introns.bed')
    if not os.path.exists(output_file):
        with open(output_file, "w") as out:
            for transcript in db.features_of_type("transcript"):
                exon_list = list(db.children(transcript, featuretype='exon', order_by='start'))
                if len(exon_list) < 2:
                    continue

                for i in range(len(exon_list) - 1):
                    first_exon = exon_list[i]
                    second_exon = exon_list[i + 1]

                    chrom = first_exon.chrom
                    strand = first_exon.strand
                    intron_start = first_exon.end
                    intron_end = second_exon.start

                    if intron_end > intron_start:
                        out.write(f"{chrom}\t{intron_start}\t{intron_end}\t.\t0\t{strand}\n")
    return output_file

def create_transcripts_bed(db):
    """
    Creates a BED file containing the full span of every transcript.
    This is used for intergenic region computation (to subtract from genome space).
    """
    output_file = os.path.join(base_dir, 'bed_files/transcripts.bed')
    if not os.path.exists(output_file):
        with open(output_file, "w") as out:
            for tx in db.features_of_type("transcript"):
                chrom = tx.chrom
                start = tx.start - 1
                end = tx.end
                strand = tx.strand
                out.write(f"{chrom}\t{start}\t{end}\t.\t0\t{strand}\n")
    return output_file

def create_intergenic_bed_file(db):
    """
   Computes intergenic regions as all genomic intervals not overlapped by any transcript.
   This replaces bedtools complement by:
   - Parsing chromosome lengths from the genome FASTA index (.fai)
   - Subtracting transcript intervals from each chromosome span
   """
    output_file = os.path.join(base_dir, 'bed_files/intergenic.bed')
    if not os.path.exists(output_file):
        chrom_sizes = {}
        with open(GENOME_FAI) as fai:
            for line in fai:
                chrom, size = line.split()[:2]
                chrom_sizes[chrom] = int(size)

        transcripts_by_chrom = defaultdict(list)

        transcripts_file = create_transcripts_bed(db)
        with open(transcripts_file) as bed:
            for line in bed:
                chrom, start, end, *_ = line.strip().split()
                chrom = chrom.removeprefix("chr")
                if chrom == "M":
                    chrom = "MT"
                transcripts_by_chrom[chrom].append((int(start), int(end)))

        with open(output_file, "w") as out:
            for chrom in chrom_sizes:
                chrom_len = chrom_sizes[chrom]
                regions = sorted(transcripts_by_chrom.get(chrom, []))

                current = 0
                for start, end in regions:
                    if start > current:
                        out.write(f"{chrom}\t{current}\t{start}\n")
                    current = max(current, end)

                if current < chrom_len:
                    out.write(f"{chrom}\t{current}\t{chrom_len}\n")
    return output_file

def get_files():
    db = get_gencode_db()
    cds_bed_file = create_cds_bed_file(db)
    five_prime_bed_file, three_prime_bed_file = create_five_three_prime_bed_files(db)
    intron_bed_file = create_intron_bed_file(db)
    intergenic_bed_file = create_intergenic_bed_file(db)

    return {
        "CDS": cds_bed_file,
        "5UTR": five_prime_bed_file,
        "3UTR": three_prime_bed_file,
        "intron": intron_bed_file,
        "intergenic": intergenic_bed_file,
    }