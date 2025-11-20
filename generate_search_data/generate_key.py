# %%
import yaml
from pathlib import Path
from cryptography.fernet import Fernet

config_path = "./config.yaml"
with open(config_path) as stream:
    config = yaml.safe_load(stream)

key_path = Path(config['key_path'])

if key_path.exists():
    print(f"Warning: Key file already exists at {key_path}")
    print("Using existing key to avoid data loss.")
else:
    key = Fernet.generate_key()
    with open(key_path, "wb") as f:
        f.write(key)
    print(f"New encryption key generated and saved to {key_path}")