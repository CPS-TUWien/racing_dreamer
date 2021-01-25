from typing import Dict
import yaml

def read_hyperparams(file: str) -> Dict:
    with open(file, 'r') as f:
        return yaml.safe_load(f)
