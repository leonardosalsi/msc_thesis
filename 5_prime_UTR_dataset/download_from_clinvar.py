import os
import tempfile

if __name__ == "__main__":
    tempdir = os.path.join(tempfile.gettempdir(), "utr_dataset_gen")
    os.makedirs(tempdir, exist_ok=True)
    d = tempfile.gettempdir()
    print(d)