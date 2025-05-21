import os
import pickle
from argparse_dataclass import dataclass, ArgumentParser
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from config import results_dir
from utils.util import print_args

VAR = False

def visualize_embeddings(model_name, embeddings_type, files):
    embeddings_out_folder = os.path.join(results_dir, 'tSNE', embeddings_type)
    os.makedirs(embeddings_out_folder, exist_ok=True)
    model_results = os.path.join(embeddings_out_folder, model_name)
    os.makedirs(model_results, exist_ok=True)
    for fpath in tqdm(files):
        layer = int(fpath.split("layer_")[-1].split(".")[0])
        tsne_path = os.path.join(model_results, f"layer_{layer}.pkl")

        with open(fpath, "rb") as f:
            data = pickle.load(f)

        embeddings = data["embeddings"]
        pca = PCA(n_components=50)
        reduced = pca.fit_transform(embeddings)
        tsne_results = TSNE(n_components=2, perplexity=40, random_state=42).fit_transform(reduced)
        with open(tsne_path, "wb") as f:
            pickle.dump(tsne_results, f)

@dataclass
class EmbConfig:
    embeddings_path: str

def extract_information(args):
    information = args.embeddings_path.rstrip('/').split('/')
    model_name = information[-1]
    embeddings_type = information[-2]
    files = sorted([os.path.join(args.embeddings_path, f) for f in os.listdir(args.embeddings_path) if f.endswith(".pkl")],
                   key=lambda f: int(f.split("layer_")[-1].split(".")[0])
                   )
    return model_name, embeddings_type, files

def parse_args():
    parser = ArgumentParser(EmbConfig)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    timestamp = print_args(args, "TSNE CALCULATION ARGUMENTS")
    model_name, embeddings_type, files = extract_information(args)
    visualize_embeddings(model_name, embeddings_type, files)