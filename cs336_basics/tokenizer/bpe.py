from cs336_basics.tokenizer.chunking import chunk, format_tokens
from cs336_basics.tokenizer.constants import TINYSTORIES_VAL, SPECIAL_TOK
from cs336_basics.tokenizer.models import BPEOutput, MergePair


VOCAB_SIZE=256

def pprint_merge_table(table: dict[tuple[bytes, ...], MergePair], title: str = "") -> None:
    """Pretty-print a merge-rank table, both by descending count and by word index."""

    def render(pair: tuple[bytes, ...]) -> str:
        return " + ".join(repr(b)[1:] for b in pair)  # drop the leading b in b'...'

    if title:
        print(f"\n{title} ({len(table)} pairs)")
    for pair, meta in sorted(table.items(), key=lambda item: item[1].count, reverse=True):
        print(f"  {render(pair):<16} count={meta.count:<5} indexes={sorted(meta.indexes)}")

    # index-wise (word-wise) distribution: which pairs live in each chunk
    by_index: dict[int, list[tuple[str, int]]] = {}
    for pair, meta in table.items():
        for idx in meta.indexes:
            by_index.setdefault(idx, []).append((render(pair), meta.count))
    for idx in sorted(by_index):
        print(f"  [index {idx}]")
        for rendered, count in sorted(by_index[idx], key=lambda item: item[1], reverse=True):
            print(f"      {rendered:<16} count={count}")


def flatten(chunks: list[dict[tuple[bytes, ...], int]]) -> None:
    chunks[:] = [
        {k: v}
        for chunk in chunks
        for k, v in chunk.items()
    ]


def get_top_merge_candidate(merge_rank_table:dict[tuple[bytes, ...], MergePair]) -> tuple[tuple[bytes, ...], MergePair]:
    max_val = max(merge_rank_table.items(), key=lambda item:(item[1].count,item[0]))
    return max_val


def merge(chunks:list[dict[tuple[bytes,...], int]],from_val:tuple[bytes, ...],meta:MergePair):
    """
    should update the chunks in place.
    should return the pairs, whose counts need to be updated.
    """

    to_val = from_val[0]+ from_val[1]
    indices = meta.indexes
    pairs_to_update: dict[tuple[bytes, ...], int] = {}
    pairs_to_add: dict[tuple[bytes, ...], MergePair] = {}
    for ind in indices:
        target = chunks[ind]
        new_cand: list[bytes]= []
        curr:list[bytes] = []
        count = 0
        update_index = 0
        for cand, v in target.items():
            curr = list(cand)
            count = v
            i = 0
            while i < len(cand):
                if i < len(cand)-1:
                    val = (cand[i], cand[i+1])
                elif i >= len(cand) - 1:
                    val = (cand[i])

                if val == from_val: #pyright:ignore
                    new_cand.append(to_val)
                    p = i
                    q = len(new_cand) - 1 

                    if p > 0:
                        left = (cand[p - 1], cand[p])
                        pairs_to_update[left] = pairs_to_update.get(left, 0) + count

                    pairs_to_update[from_val] = pairs_to_update.get(from_val, 0) + count

                    if p + 2 < len(cand):
                        right = (cand[p + 1], cand[p + 2])
                        pairs_to_update[right] = pairs_to_update.get(right, 0) + count

                    if q > 0:
                        a = (new_cand[q - 1], to_val)
                        prev = pairs_to_add.get(a)
                        if prev:
                            prev.count += count
                            prev.indexes.add(ind)
                        else:
                            pairs_to_add[a] = MergePair(count=count, indexes={ind})

                    if p + 2 < len(cand):
                        a = (to_val, cand[p + 2])
                        prev = pairs_to_add.get(a)
                        if prev:
                            prev.count += count
                            prev.indexes.add(ind)
                        else:
                            pairs_to_add[a] = MergePair(count=count, indexes={ind})
                    i += 2
                else:
                    new_cand.append(cand[i])
                    i += 1

            chunks[ind] = {tuple(new_cand): v}

    return pairs_to_add, pairs_to_update




def train_bpe(input_path:str, vocab_size:int, special_tokens:list[str]):
    chunks = chunk(file=input_path)
    curr_id = VOCAB_SIZE 

    vocabulary = {i:bytes([i]) for i in range(VOCAB_SIZE)}
    # contains the ranking of the top two byte pairs, will keep on changing.
    merge_rank_table: dict[tuple[bytes, ...], MergePair] = {}


    num_merges = vocab_size - 256 - len(special_tokens)
    merge_table: list[tuple[bytes, bytes]] = []
    flatten(chunks)

    for ind, chnk in enumerate(chunks):
        for k, v in chnk.items():
            i = 0
            for l in range(len(k)-1):
                val = (k[l], k[l+1])
                prev = merge_rank_table.get(val, None)
                if prev:
                    merge_rank_table[val].count += v
                    merge_rank_table[val].indexes.add(ind)
                else:
                    merge_rank_table[val] = MergePair(count=v, indexes=set([ind]))

    for _ in range(num_merges):
        top_pair = get_top_merge_candidate(merge_rank_table)
        merge_table.append(top_pair[0]) #pyright:ignore

        vocabulary[curr_id] = top_pair[0][0] + top_pair[0][1]
        curr_id += 1

        pairs_to_add, pairs_to_update = merge(chunks, top_pair[0], top_pair[1])

        for k, v in pairs_to_add.items():
            existing_val = merge_rank_table.get(k, None)
            if existing_val:
                merge_rank_table[k] = MergePair(count=v.count + existing_val.count, indexes=v.indexes | existing_val.indexes)
            else:
                merge_rank_table[k] = MergePair(count = v.count, indexes=v.indexes)

        for k,v in pairs_to_update.items():
            old = merge_rank_table.get(k)
            if old:
                new_count = old.count - v

                if new_count > 0:
                    merge_rank_table[k] = MergePair(count=old.count - v, indexes=old.indexes)

                else:
                    del merge_rank_table[k]


    for tok in special_tokens:
        vocabulary[curr_id] = tok.encode("utf-8")
        curr_id += 1

    # return merge_rank_table
    return BPEOutput(vocabulary=vocabulary, merges=merge_table)




if __name__=="__main__":


#     t1 = {
#     (b"a", b"b"): MergePair(count=5, indexes={0}),
#     (b"a", b"c"): MergePair(count=7, indexes={0}),
#     (b"z", b"a"): MergePair(count=7, indexes={1}),   # ties with (b"a", b"c")
#     (b"h", b"i"): MergePair(count=2, indexes={0}),
# }

#     print(get_top_merge_candidate(t1))

#     t2 = {
#     (b"t", b"a"): MergePair(count=4, indexes={0}),
#     (b"t", b"h"): MergePair(count=4, indexes={0}),   # ties with (b"t", b"a")
#     (b"t", b"e"): MergePair(count=1, indexes={0}),
# }

#     print(get_top_merge_candidate(t2))
#     # train_bpe(TINYSTORIES_VAL, 255, [])

#     t3 = {
#     (b"th", b"e"): MergePair(count=6, indexes={0}),
#     (b"t", b"he"): MergePair(count=6, indexes={1}),  # ties with (b"th", b"e")
# }

#     print(get_top_merge_candidate(t3))


    # A — merge in the middle 
    chunks_a = [{(b"l", b"o", b"w"): 5}] 
    print(merge(chunks_a, (b"l", b"o"), MergePair(count=5, indexes={0}))) 
    assert chunks_a            == [{(b"lo", b"w"): 5}] 
    # B — merge at the end (no right neighbor) 
    chunks_b = [{(b"l", b"o", b"w"): 5}] 
    print(merge(chunks_b, (b"o", b"w"), MergePair(count=5, indexes={0}))) 
    assert chunks_b ==   [{(b"l", b"ow"): 5}] 
    # C — both neighbors present (start + end branches both fire) 
    chunks_c = [{(b"f", b"l", b"o", b"w"): 5}]
    print(merge(chunks_c, (b"l", b"o"), MergePair(count=5, indexes={0})))
    assert chunks_c == [{(b"f", b"lo", b"w"): 5}]

    # # D — overlapping pair (the trap): greedy left-to-right => ONE merge
    chunks_d = [{(b"a", b"a", b"a"): 1}]
    print(merge(chunks_d, (b"a", b"a"), MergePair(count=1, indexes={0})))
    assert chunks_d == [{(b"aa", b"a"): 1}]

    # E — two non-overlapping occurrences in one word
    chunks_e = [{(b"a", b"b", b"a", b"b"): 1}]
    print(merge(chunks_e, (b"a", b"b"), MergePair(count=1, indexes={0})))
    assert chunks_e == [{(b"ab", b"ab"): 1}]

    # F — multiple chunks; chunk 1 also contains the pair but is NOT listed
    chunks_f = [{(b"l", b"o", b"w"): 5}, {(b"l", b"o", b"g"): 2}, {(b"l", b"o", b"w"): 3}]
    print(merge(chunks_f, (b"l", b"o"), MergePair(count=8, indexes={0, 2})))
    assert chunks_f == [{(b"lo", b"w"): 5}, {(b"l", b"o", b"g"): 2}, {(b"lo", b"w"): 3}]

    # G — pair absent in a listed chunk
    chunks_g = [{(b"c", b"a", b"t"): 4}]
    print(merge(chunks_g, (b"l", b"o"), MergePair(count=0, indexes={0})))
    assert chunks_g == [{(b"c", b"a", b"t"): 4}]

    # H — longer overlap run
    chunks_h = [{(b"a", b"a", b"a", b"a"): 1}]
    print(merge(chunks_h, (b"a", b"a"), MergePair(count=1, indexes={0})))
    assert chunks_h == [{(b"aa", b"aa"): 1}]

    # I — odd-length overlap run
    chunks_i = [{(b"a", b"a", b"a", b"a", b"a"): 1}]
    print(merge(chunks_i, (b"a", b"a"), MergePair(count=1, indexes={0})))
    assert chunks_i == [{(b"aa", b"aa", b"a"): 1}]

    # J — same neighbor pair occurs twice (probes the RETURN, not chunks)
    chunks_j = [{(b"x", b"a", b"b", b"x", b"a", b"b"): 1}]
    print(merge(chunks_j, (b"a", b"b"), MergePair(count=1, indexes={0})))
    assert chunks_j == [{(b"x", b"ab", b"x", b"ab"): 1}]


    # chunks_sample = [{(b"l", b"o", b"w"): 5}, {(b"l", b"o", b"w", b"e", b"r"): 2}, {(b"w", b"i", b"d", b"e", b"s", b"t"): 3}, {(b"n", b"e", b"w", b"e", b"s", b"t"):6}]
    # sample_path = "/Users/unnat-deepsource/personal/cs336/assignment1-basics/data/sample.txt"
    # new = train_bpe(sample_path, 0, [])
    # pprint_merge_table(new, "after merge")
