import numpy as np
import random

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
    
    return arr


def predictions_to_string(arr: np.ndarray, num_examples: int = 10) -> str:
    """
    Expect a numpy array with shape [#batch_size, length, #vocab_size]
    
    Convert a 2D NumPy array of character IDs back to a string.
    Parameters
    ----------
        
    """
    vocab_ids = np.argmax(arr, axis=-1)
    
    results = []
    
    examples = random.choices(range(vocab_ids.shape[0]), k=num_examples) if num_examples < vocab_ids.shape[0] else list(range(vocab_ids.shape[0]))
    
    for i in examples:
        example = vocab_ids[i]
        # Convert IDs back to characters
        chars = [ID_TO_CHAR.get(id_, '?') for id_ in example]
        math_expr = ''.join(chars).rstrip()
        results.append(f"Example {i}: {math_expr}")
    return "\n".join(results)


if __name__ == "__main__":
    # Example usage
    example_string = "3+5=8"
    fixed_length = 10
    arr = string_to_fixed_array(example_string, length=fixed_length)
    print("Fixed-length array:", arr)
    
    # simulate predictions
    predictions = np.zeros((1, fixed_length, len(CHAR_MAPPING)))  #
    
    for idx, char in enumerate(example_string):
        if char in CHAR_MAPPING:
            predictions[0, idx, CHAR_MAPPING[char]] = 1.0
            
    # Convert back to string
    reconstructed_string = predictions_to_string(predictions)
    print("Reconstructed string:", reconstructed_string)
    
