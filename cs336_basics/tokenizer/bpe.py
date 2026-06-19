from cs336_basics.tokenizer.chunking import chunk, format_tokens
from cs336_basics.tokenizer.constants import TINYSTORIES_VAL, SPECIAL_TOK
from cs336_basics.tokenizer.models import BPEOutput, MergePair


def flatten(chunks: list[dict[tuple[bytes, ...], int]]) -> None:
    chunks[:] = [
        {k: v}
        for chunk in chunks
        for k, v in chunk.items()
    ]


def get_top_merge_candidate(merge_rank_table:dict[tuple[bytes, ...], MergePair]) -> tuple[tuple[bytes, ...], MergePair]:

    # ranked = dict(sorted(merge_rank_table.items(), key=lambda item:item[1].count, reverse=True))

    max_val = max(merge_rank_table.items(), key=lambda item:(item[0],item[1].count))
    return max_val


def merge(chunks:list[dict[tuple[bytes,...], int]],from_val:tuple[bytes], to_val:tuple[bytes], meta:MergePair):
    """
    should update the chunks in place.
    should return the pairs, whose counts need to be updated.
    """
    indices = meta.indexes 
    pairs_to_update: list[tuple[bytes, ...]] = []
    for ind in indices: 
        target = chunks[ind]
        cand = target.keys()
        new_cand: list[bytes]= []
        for cand, v in target.items():
            for i in range(len(cand)):
                val = (cand[i], cand[i+1])
                if from_val == val:
                    if i-1 >= 0:
                        pairs_to_update.append((cand[i-1], cand[i]))    

                    if i+2 < len(cand):
                        pairs_to_update.append((cand[i+1], cand[i+2]))
                        i = i+2
                    new_cand.append(to_val[0])
                else:
                    new_cand.append(cand[i])
        
            chunks[ind] = {tuple(new_cand): v}
                    



def train_bpe(input_path:str, vocab_size:int, special_tokens:list[str]) -> BPEOutput:
    chunks = chunk(file=input_path)
    # contains the ranking of the top two byte pairs, will keep on changing.
    merge_rank_table: dict[tuple[bytes, ...], MergePair] = {}

    merge_table: list[tuple[bytes, ...]] = []
    # chunks = [{(b"l", b"o", b"w"): 3}, {(b"l", b"o", b"w", b"e", b"r"): 2}] 
    # chunks[0] = {(b"l",b"o",b"w"): 3, (b"l",b"o",b"g"): 1}

    flatten(chunks)
    for ind, chnk in enumerate(chunks):
        for k, v in chnk.items():
            for l in range(len(k)-1):
                val = (k[l], k[l+1])
                prev = merge_rank_table.get(val, None)
                if prev:
                    merge_rank_table[val].count += v 
                    merge_rank_table[val].indexes.add(ind)
                else:
                    merge_rank_table[val] = MergePair(count=v, indexes=set([ind]))


    top_pair = get_top_merge_candidate(merge_rank_table)
    merge_table.append(top_pair[0])

    index_info = top_pair[1].indexes



    return BPEOutput(vocabulary={}, merges=[])




if __name__=="__main__":
    t1 = {
    (b"a", b"b"): MergePair(count=5, indexes={0}),
    (b"a", b"c"): MergePair(count=7, indexes={0}),
    (b"z", b"a"): MergePair(count=7, indexes={1}),   # ties with (b"a", b"c")
    (b"h", b"i"): MergePair(count=2, indexes={0}),
}
    
    print(get_top_merge_candidate(t1))

    t2 = {
    (b"t", b"a"): MergePair(count=4, indexes={0}),
    (b"t", b"h"): MergePair(count=4, indexes={0}),   # ties with (b"t", b"a")
    (b"t", b"e"): MergePair(count=1, indexes={0}),
}
    
    print(get_top_merge_candidate(t2))
    # train_bpe(TINYSTORIES_VAL, 255, [])

    t3 = {
    (b"th", b"e"): MergePair(count=6, indexes={0}),
    (b"t", b"he"): MergePair(count=6, indexes={1}),  # ties with (b"th", b"e")
}
    
    print(get_top_merge_candidate(t3))


