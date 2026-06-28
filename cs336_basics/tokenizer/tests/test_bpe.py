"""
Standalone tests for the BPE `merge` bookkeeping.
Run with:  pytest test_merge.py -v
"""
from collections import Counter

import pytest

from cs336_basics.tokenizer.bpe import merge
from cs336_basics.tokenizer.models import MergePair


# ---- helpers ---------------------------------------------------------------

def one(word, count):
    """A single-pretoken chunk list."""
    return [{tuple(word): count}]


def bigram_counts(chunks):
    """Reference: recompute every bigram count from scratch over `chunks`."""
    c = Counter()
    for d in chunks:
        for word, freq in d.items():
            for pair in zip(word, word[1:]):
                c[pair] += freq
    return c


def norm_add(pairs_to_add):
    """MergePair may not define __eq__; compare on (count, indexes)."""
    return {k: (v.count, frozenset(v.indexes)) for k, v in pairs_to_add.items()}


# ---- rewrite of chunks (A–J, K) -------------------------------------------

@pytest.mark.parametrize(
    "word, count, from_val, expected",
    [
        ((b"l", b"o", b"w"),                5, (b"l", b"o"), (b"lo", b"w")),            # A middle
        ((b"l", b"o", b"w"),                5, (b"o", b"w"), (b"l", b"ow")),            # B end
        ((b"f", b"l", b"o", b"w"),          5, (b"l", b"o"), (b"f", b"lo", b"w")),     # C both sides
        ((b"a", b"a", b"a"),                1, (b"a", b"a"), (b"aa", b"a")),           # D overlap
        ((b"a", b"b", b"a", b"b"),          1, (b"a", b"b"), (b"ab", b"ab")),         # E twice
        ((b"c", b"a", b"t"),                4, (b"l", b"o"), (b"c", b"a", b"t")),     # G absent
        ((b"a", b"a", b"a", b"a"),          1, (b"a", b"a"), (b"aa", b"aa")),         # H run
        ((b"a", b"a", b"a", b"a", b"a"),    1, (b"a", b"a"), (b"aa", b"aa", b"a")),   # I odd run
        ((b"x", b"a", b"b", b"x", b"a", b"b"), 1, (b"a", b"b"), (b"x", b"ab", b"x", b"ab")),  # J
        ((b"a", b"b", b"c", b"d", b"b", b"c"), 1, (b"a", b"b"), (b"ab", b"c", b"d", b"b", b"c")),  # K
    ],
)
def test_rewrite(word, count, from_val, expected):
    chunks = one(word, count)
    merge(chunks, from_val, MergePair(count=count, indexes={0}))
    assert chunks == [{expected: count}]


def test_rewrite_multichunk():  # F — only listed indexes are touched
    chunks = [{(b"l", b"o", b"w"): 5}, {(b"l", b"o", b"g"): 2}, {(b"l", b"o", b"w"): 3}]
    merge(chunks, (b"l", b"o"), MergePair(count=8, indexes={0, 2}))
    assert chunks == [{(b"lo", b"w"): 5}, {(b"l", b"o", b"g"): 2}, {(b"lo", b"w"): 3}]


# ---- returned count deltas (E, J, K) --------------------------------------

def test_return_E():
    chunks = one((b"a", b"b", b"a", b"b"), 1)
    add, update = merge(chunks, (b"a", b"b"), MergePair(count=1, indexes={0}))
    assert update == {(b"a", b"b"): 2, (b"b", b"a"): 1}
    assert norm_add(add) == {(b"ab", b"ab"): (1, frozenset({0}))}


def test_return_J():
    chunks = one((b"x", b"a", b"b", b"x", b"a", b"b"), 1)
    add, update = merge(chunks, (b"a", b"b"), MergePair(count=1, indexes={0}))
    assert update == {(b"x", b"a"): 2, (b"a", b"b"): 2, (b"b", b"x"): 1}
    assert norm_add(add) == {(b"x", b"ab"): (2, frozenset({0})),
                             (b"ab", b"x"): (1, frozenset({0}))}


def test_return_K_partial_survival():
    chunks = one((b"a", b"b", b"c", b"d", b"b", b"c"), 1)
    add, update = merge(chunks, (b"a", b"b"), MergePair(count=1, indexes={0}))
    assert update == {(b"a", b"b"): 1, (b"b", b"c"): 1}   # (b,c): 2 copies, 1 destroyed
    assert norm_add(add) == {(b"ab", b"c"): (1, frozenset({0}))}


# ---- the general invariant: deltas must reconcile old -> new ---------------

WORDS = [
    (b"a", b"b", b"a", b"b"),
    (b"x", b"a", b"b", b"x", b"a", b"b"),
    (b"a", b"b", b"c", b"d", b"b", b"c"),
    (b"a", b"a", b"a"),
    (b"a", b"a", b"a", b"a"),
    (b"a", b"a", b"a", b"a", b"a"),
    (b"b", b"a", b"n", b"a", b"n", b"a"),
    (b"c", b"a", b"t"),            # from_val absent -> must be a no-op
]
FROM_VALS = [(b"a", b"b"), (b"a", b"a"), (b"a", b"n"), (b"n", b"a")]


@pytest.mark.parametrize("word", WORDS)
@pytest.mark.parametrize("from_val", FROM_VALS)
def test_deltas_reconcile(word, from_val):
    freq = 3
    chunks = one(word, freq)
    old = bigram_counts(one(word, freq))          # ground truth before
    add, update = merge(chunks, from_val, MergePair(count=freq, indexes={0}))
    new = bigram_counts(chunks)                    # ground truth after (mutated in place)

    for k in set(old) | set(new) | set(update) | set(add):
        got = old.get(k, 0) - update.get(k, 0) + (add[k].count if k in add else 0)
        assert got == new.get(k, 0), (
            f"{k}: old={old.get(k,0)} -dec={update.get(k,0)} "
            f"+inc={add[k].count if k in add else 0} => {got}, want {new.get(k,0)}"
        )
