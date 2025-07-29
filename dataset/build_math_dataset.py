import numpy as np
import os
import json

from random import randint
from typing import Optional
from argdantic import ArgParser
from pydantic import BaseModel
from common import PuzzleDatasetMetadata

cli = ArgParser()

class DataProcessConfig(BaseModel):
    output_dir: str = "data/math-100-simple"
    num_digits: int = 4
    num_samples: int = 5000

    subsample_size: Optional[int] = None
    aug: bool = False

import numpy as np

# -----------------------------------------------------------------------------
# Character-level mapping: each character (digit, sign, space) maps to a unique ID
# -----------------------------------------------------------------------------
CHAR_MAPPING = {
    ' ': 0,  # blank / padding
    '0': 1, '1': 2, '2': 3, '3': 4, '4': 5,
    '5': 6, '6': 7, '7': 8, '8': 9, '9': 10,
    '+': 11, '-': 12, '*': 13, '/': 14, '=': 15
}

# Optional: reverse mapping for decoding
ID_TO_CHAR = {v: k for k, v in CHAR_MAPPING.items()}


def string_to_fixed_array(s: str, length: int = 100) -> np.ndarray:
    """
    Convert a string into a fixed‐length NumPy array of character IDs.
    
    Parameters
    ----------
    s : str
        The input string, comprised of digits, operators, '=', or spaces.
    length : int, optional (default=100)
        The desired fixed length of the output array. Strings shorter than this
        are padded with the ID for space; longer strings raise an error.
    
    Returns
    -------
    np.ndarray of shape (length,)
        Array of dtype int32, where each element is the ID corresponding to the
        character in the mapping. Padding is applied at the end for shorter strings.
    
    Raises
    ------
    ValueError
        If `len(s) > length` or if an unsupported character is encountered.
    """
    if len(s) > length:
        raise ValueError(f"Input string length ({len(s)}) exceeds fixed length {length}.")
    
    arr = np.full(shape=(length,), fill_value=CHAR_MAPPING[' '], dtype=np.int32)
    
    for idx, ch in enumerate(s):
        if ch not in CHAR_MAPPING:
            raise ValueError(f"Unsupported character '{ch}' at position {idx}.")
        arr[idx] = CHAR_MAPPING[ch]
    
    print(f"Converted string '{s}' to fixed-length array: {arr}")
    
    return arr


def generate_dataset(set_name: str, config: DataProcessConfig):
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0
    
    max_sequence_length = 3 * config.num_digits + 5  # Maximum length of the input/output sequencs
    
    for _ in range(config.num_samples):
        # Generate a simple math problem
        a = randint(0, 10**config.num_digits - 1)
        b = randint(0, 10**config.num_digits - 1)
        operation = '+' if randint(0, 1) == 0 else '-'
        
        if operation == '+':
            result = a + b
        else:
            result = a - b
        
        # Format the problem and solution
        problem = f"{a}{operation}{b}="
        problem_with_solution = f"{a}{operation}{b}={result}"
        
        # Convert to fixed-length arrays
        input_array = string_to_fixed_array(problem, length=max_sequence_length)
        label_array = string_to_fixed_array(problem_with_solution, length=max_sequence_length)
        results["inputs"].append(input_array)
        results["labels"].append(label_array)
        results["puzzle_identifiers"].append(0)
        results["puzzle_indices"].append(example_id)
        results["group_indices"].append(puzzle_id)
        example_id += 1
        puzzle_id += 1
        
    # Convert lists to numpy arrays
    np_results = {
        "inputs": np.array(results["inputs"], dtype=np.int32),
        "labels": np.array(results["labels"], dtype=np.int32),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=int(max_sequence_length),  # type: ignore
        vocab_size=len(CHAR_MAPPING),  # PAD + Charset
        
        pad_id=0,
        ignore_label_id=-100,
        
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        sets=["all"]
    )
    
    # Save metadata as JSON.
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
        
    # Save data
    for k, v in np_results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
        
    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    generate_dataset("train", config)
    generate_dataset("test", config)


if __name__ == "__main__":
    cli()
