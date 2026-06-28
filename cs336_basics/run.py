from cs336_basics.tokenizer.bpe import train_bpe
from cs336_basics.tokenizer.constants import TINYSTORIES_TRAIN, TINY_STORIES_TRAIN_RESULT


if __name__ == "__main__":
    # out = train_bpe(TINYSTORIES_TRAIN, 10000, ["<|endoftext|>"])
    # vocab = out.vocabulary
    # merges = out.merges
    import pickle
    with open(TINY_STORIES_TRAIN_RESULT, 'rb') as f:
        result = pickle.load(f)
        print(result)
    # resfile_vocab = open(TINY_STORIES_TRAIN_RESULT, "wb")
    # pickle.dump(vocab, resfile_vocab)
    # resfile_vocab.close()



