import asyncio
from pprint import pprint

import pandas as pd
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport

GENES = [
    "BRCA1",   # Breast cancer susceptibility gene 1
    "BRCA2",   # Breast cancer susceptibility gene 2
    "TP53",    # Tumor suppressor p53
    "PTEN",    # Tumor suppressor PTEN
    "CFTR",    # Cystic fibrosis gene
    "ATF4",    # Stress-response transcription factor
    "DDIT3",   # ER-stress gene, regulated by uORFs
    "EIF2AK4", # Translation initiation control (GCN2)
    "CDKN1B",  # Cell cycle gene, known UTR regulation
    "FMR1",    # Fragile X gene
    "ACTB",    # Housekeeping gene
    "GAPDH",   # Housekeeping gene
    "RPL13A",  # Ribosomal protein
    "RPS6",    # Ribosomal protein
    "MYC"      # Oncogene, UTR tightly regulated
]

async def query_gene_variants(gene):
    transport = AIOHTTPTransport(url="https://gnomad.broadinstitute.org/api")
    client = Client(transport=transport, fetch_schema_from_transport=True)

    query = gql("""
        query GetVariants($gene: String!) {
          gene(gene_symbol: $gene, reference_genome: GRCh38) {
            variants(dataset: gnomad_r4) {
              variant_id
              pos
              consequence
              exome {
                af
              }
            }
          }
        }
    """)

    try:
        result = await client.execute_async(query, variable_values={"gene": gene})
        records = []
        for var in result["gene"]["variants"]:
            pprint(var)
            if var["exome"] is None:
                continue
            af = var["exome"]["af"]
            if af and af > 0.01 and "5_prime_UTR_variant" in var["consequence"]:
                records.append({
                    "gene": gene,
                    "variant_id": var["variant_id"],
                    "position": var["pos"],
                    "af": af,
                    "consequence": var["consequence"]
                })
        return records
    except Exception as e:
        print(f"Error querying {gene}: {e}")
        return []

async def main():
    all_records = []
    for gene in GENES:
        print(f"Querying {gene}...")
        results = await query_gene_variants(gene)
        all_records.extend(results)

    df = pd.DataFrame(all_records)
    out_path = "gnomad_5utr_benign_variants.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} likely benign 5′UTR variants to {out_path}")

if __name__ == "__main__":
    asyncio.run(main())
