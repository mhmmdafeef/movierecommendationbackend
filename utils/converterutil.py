import numpy as np

def convert_pg_array_string_to_numpy(s: str) -> np.ndarray:
    return np.array(s.strip('{}').split(','), dtype=np.float32)
