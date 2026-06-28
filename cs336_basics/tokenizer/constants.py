
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
SPECIAL_TOK = b"<|endoftext|>"

TINYSTORIES_TRAIN = "/Users/unnat-deepsource/personal/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
TINYSTORIES_VAL = "/Users/unnat-deepsource/personal/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"

TINY_STORIES_TRAIN_RESULT= "/Users/unnat-deepsource/personal/cs336/assignment1-basics/data/TS_train_res.pkl"

NUM_CHUNKS = 20