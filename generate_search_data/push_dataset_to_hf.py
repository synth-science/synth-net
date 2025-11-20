# %%
import pickle
import yaml
import pandas as pd
from pathlib import Path
from cryptography.fernet import Fernet
from datasets import Dataset

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
df = pd.read_parquet(config["save_centroid_search_data_path"])
# %%
encrypted_rows = []
for _, row in df.iterrows():
    encrypted_row = {
        col: cipher.encrypt(pickle.dumps(row[col])) 
        for col in df.columns
    }
    encrypted_rows.append(encrypted_row)

# %%
dataset = Dataset.from_list(encrypted_rows)
dataset.push_to_hub(
    "magnolia-psychometrics/apa-psyc-tests",
    private=True
)