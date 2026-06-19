from typing import BinaryIO 
import os
from cs336_basics.tokenizer.constants import SPECIAL_TOK, NUM_CHUNKS, PAT
from models import BPEOutput
from multiprocessing import Pool
import regex as re
from functools import partial

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))



def pretokenize(start: int, end:int,fp:str,) -> dict[tuple[bytes,...], int]:

    pt = []
    with open(fp, "rb") as f:
        f.seek(start) 
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        chunks = chunk.split(str(SPECIAL_TOK.decode('utf-8')))
        for chunk in chunks:
            for m in re.finditer(PAT, chunk):
                pt.append(m.group())

    freq: dict[str, int] = {}

    # for cand in pt:
    for word in pt:
        freq[word] = freq.get(word, 0) + 1

    return format_tokens(freq)

def chunk(file:str, num_chunks: int = NUM_CHUNKS, special_tok: bytes = SPECIAL_TOK) -> list[dict[tuple[bytes, ...], int]]:
    with open(file, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, special_tok)

        chunk_steps:list[tuple[int,int]] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            chunk_steps.append((start, end)) 
         

    with Pool() as pool:
        fn = partial(pretokenize, fp=file)
        results = pool.starmap(fn, chunk_steps)

    return results


def format_tokens(toks:dict[str, int]) -> dict[tuple[bytes, ...], int]:
    formatted_toks: dict[tuple[bytes, ...], int] = {}

    for k, v in toks.items():
        encoded_k = k.encode("utf-8")
        key = [tuple([encoded_k[i:i+1] for i in range(len(encoded_k))]) ]
        formatted_toks[key[0]] = v

    return formatted_toks

