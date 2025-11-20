# %%
import pickle
import yaml
import pandas as pd
from pathlib import Path
from cryptography.fernet import Fernet
from datasets import load_dataset

# %%
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:    
    SCRIPT_DIR = Path.cwd()

# %%
config_path = f"{SCRIPT_DIR}/config.yaml"

with open(config_path) as stream:
    config = yaml.safe_load(stream)

key_path = config['key_path']
with open(key_path, "rb") as f:
    key = f.read()

cipher = Fernet(key)

# %%
dataset = load_dataset("magnolia-psychometrics/apa-psyc-tests", split="train")

decrypted_rows = []
for row in dataset:
    decrypted_row = {
        col: pickle.loads(cipher.decrypt(row[col]))
        for col in row.keys()
    }
    decrypted_rows.append(decrypted_row)

# %%
centroid_search_data = pd.DataFrame(decrypted_rows)
# %%
item_search_data = pd.read_parquet(path=config['save_item_search_data_path'])