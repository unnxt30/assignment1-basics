"""
Standalone tests for the IndexedHeap (a max-heap keyed by `pair`).
Run with:  pytest cs336_basics/tokenizer/tests/test_index_heap.py -v

The HeapNode order is a MAX order on (count, pair):  larger count wins, ties
broken by the larger pair.  So `sorted(nodes, reverse=True)` is the exact
drain order pop_max should produce.

Two structural invariants are checked after *every* mutation, the same way
test_bpe.py recomputes bigram counts from scratch as ground truth:
  - the heap property  (each parent dominates its children)
  - index consistency  (h.index[pair] is the true slot of that pair, and the
                        map covers exactly the heap)
"""
import random

import pytest

from cs336_basics.tokenizer.models import HeapNode, IndexedHeap


# ---- helpers ---------------------------------------------------------------

def node(pair, count):
    return HeapNode(count=count, pair=pair)


def assert_heap_property(h: IndexedHeap):
    """Every parent must dominate its children under the HeapNode max order."""
    n = len(h.heap)
    for i in range(n):
        l, r = 2 * i + 1, 2 * i + 2
        if l < n:
            assert not (h.heap[i] < h.heap[l]), \
                f"heap property broken: parent {i} < left child {l}"
        if r < n:
            assert not (h.heap[i] < h.heap[r]), \
                f"heap property broken: parent {i} < right child {r}"


def assert_index_consistent(h: IndexedHeap):
    """index must map each pair to its real slot and cover exactly the heap."""
    assert len(h.index) == len(h.heap), \
        f"index size {len(h.index)} != heap size {len(h.heap)}"
    for i, nd in enumerate(h.heap):
        assert nd.pair in h.index, f"pair {nd.pair} at slot {i} missing from index"
        assert h.index[nd.pair] == i, \
            f"index[{nd.pair}]={h.index[nd.pair]} but it actually lives at {i}"


def build(nodes):
    """Push a list of nodes, asserting both invariants after each push."""
    h = IndexedHeap()
    for nd in nodes:
        h.push(nd)
        assert_heap_property(h)
        assert_index_consistent(h)
    return h


def as_keys(nodes):
    """Comparable, order-revealing view of a node sequence."""
    return [(n.count, n.pair) for n in nodes]


# ---- push: index bookkeeping & ordering ------------------------------------

def test_push_single_registers_index():
    """A lone pushed node must still be findable in the index map."""
    h = IndexedHeap()
    h.push(node((b"a", b"b"), 5))
    assert_index_consistent(h)            # catches "push never records the slot"


@pytest.mark.parametrize(
    "counts",
    [
        [3],
        [1, 2, 3, 4, 5],          # ascending -> every push bubbles to the root
        [5, 4, 3, 2, 1],          # descending -> nothing moves
        [3, 1, 4, 1, 5, 9, 2, 6], # mixed
        [7, 7, 7, 7],             # all-equal counts -> pair breaks the tie
    ],
)
def test_push_keeps_invariants(counts):
    nodes = [node((bytes([i]), b"x"), c) for i, c in enumerate(counts)]
    build(nodes)                          # invariants asserted inside build()


def test_push_puts_global_max_at_root():
    h = build([node((bytes([i]), b"x"), c) for i, c in enumerate([3, 1, 4, 1, 5, 9, 2])])
    top = max(h.heap, key=lambda n: (n.count, n.pair))
    assert (h.heap[0].count, h.heap[0].pair) == (top.count, top.pair)


def test_push_tie_breaks_on_pair():
    """Equal counts: the larger pair is the dominant one and lands at the root."""
    h = build([node((b"a",), 5), node((b"z",), 5), node((b"m",), 5)])
    assert h.heap[0].pair == (b"z",)


# ---- update: sift up on increase, sift down on decrease --------------------

def test_update_increase_bubbles_up():
    h = build([node((bytes([i]), b"x"), c) for i, c in enumerate([9, 8, 7, 1, 2, 3])])
    h.update(node((bytes([3]), b"x"), 0), delta=+50)   # count field is ignored; pair locates it
    assert_heap_property(h)
    assert_index_consistent(h)
    assert h.heap[0].pair == (bytes([3]), b"x")         # it now dominates everything

def test_update_decrease_sinks_down():
    h = build([node((bytes([i]), b"x"), c) for i, c in enumerate([9, 8, 7, 1, 2, 3])])
    h.update(node((bytes([0]), b"x"), 0), delta=-50)    # demote the current root
    assert_heap_property(h)
    assert_index_consistent(h)
    assert h.heap[0].pair != (bytes([0]), b"x")


def test_update_reflects_new_count():
    h = build([node((b"a",), 5), node((b"b",), 4)])
    h.update(node((b"b",), 0), delta=+3)
    assert h.heap[h.index[(b"b",)]].count == 7


# ---- pop_max: PEEK the dominant node; the heap is NOT mutated --------------
# Contract: pop_max returns the current max and leaves the heap untouched.
# Removal is a separate operation (delete()).

def test_pop_max_peeks_without_removing():
    h = build([node((b"a",), 1), node((b"b",), 9), node((b"c",), 5)])
    size = len(h.heap)
    top = h.pop_max()
    assert (top.count, top.pair) == (9, (b"b",))
    assert len(h.heap) == size             # peek must not shrink the heap
    assert (b"b",) in h.index
    assert_heap_property(h)
    assert_index_consistent(h)


def test_pop_max_is_idempotent():
    """Peeking twice yields the same node and never mutates the heap."""
    h = build([node((b"a",), 1), node((b"b",), 9), node((b"c",), 5)])
    first, second = h.pop_max(), h.pop_max()
    assert (first.count, first.pair) == (second.count, second.pair)
    assert len(h.heap) == 3


@pytest.mark.parametrize("seed", range(6))
def test_pop_max_is_always_the_global_max(seed):
    """
    Reconcile test (no delete needed): drive the heap through a random mix of
    pushes and updates, track ground-truth counts in a plain dict, and after
    every step demand that pop_max equals the true maximum — while the heap
    size never changes, proving it stays a pure peek.
    """
    rng = random.Random(seed)
    pairs = [(bytes([i]), bytes([j])) for i in range(4) for j in range(4)]
    counts = {p: rng.randint(1, 20) for p in pairs}

    h = IndexedHeap()
    for p in pairs:
        h.push(node(p, counts[p]))
        assert_heap_property(h)
        assert_index_consistent(h)

    for _ in range(40):
        p = rng.choice(pairs)
        delta = rng.randint(-8, 8)
        h.update(node(p, 0), delta)          # count ignored; pair is the key
        counts[p] += delta
        assert_heap_property(h)
        assert_index_consistent(h)
        assert h.heap[h.index[p]].count == counts[p]

        best_pair, best_count = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
        top = h.pop_max()
        assert (top.count, top.pair) == (best_count, best_pair)
        assert len(h.heap) == len(pairs)     # still a pure peek


# ---- delete: remove a specific node, located by its pair -------------------
# Contract: delete(node) removes the node whose pair == node.pair, leaving a
# valid heap and a consistent index (the count field is ignored, like update).

def test_delete_the_max():
    h = build([node((b"a", b"x"), 1), node((b"b", b"x"), 9), node((b"c", b"x"), 5)])
    h.delete(node((b"b", b"x"), 0))            # remove the current root/max
    assert (b"b", b"x") not in h.index
    assert len(h.heap) == 2
    assert_heap_property(h)
    assert_index_consistent(h)
    top = h.pop_max()
    assert (top.count, top.pair) == (5, (b"c", b"x"))   # next max surfaced


def test_delete_a_non_root_node():
    h = build([node((b"a", b"x"), 1), node((b"b", b"x"), 9), node((b"c", b"x"), 5)])
    h.delete(node((b"a", b"x"), 0))            # remove a leaf, not the max
    assert (b"a", b"x") not in h.index
    assert len(h.heap) == 2
    assert_heap_property(h)
    assert_index_consistent(h)


@pytest.mark.parametrize("seed", range(6))
def test_delete_in_random_order_keeps_invariants(seed):
    """Delete every pair in a random order; the heap stays valid throughout."""
    rng = random.Random(seed)
    pairs = [(bytes([i]), bytes([j])) for i in range(4) for j in range(4)]
    h = build([node(p, rng.randint(1, 20)) for p in pairs])

    order = pairs[:]
    rng.shuffle(order)
    for p in order:
        h.delete(node(p, 0))
        assert_heap_property(h)
        assert_index_consistent(h)
    assert h.heap == []
    assert h.index == {}


def drain(h: IndexedHeap):
    """Read the max (pop_max, a peek) then remove it (delete), until empty."""
    out = []
    while len(h.heap) > 0:
        top = h.pop_max()
        out.append(top)
        h.delete(top)
        assert_heap_property(h)
        assert_index_consistent(h)
    return out


# ---- the general invariant: push + update + drain reconciles with sorted() --

@pytest.mark.parametrize("seed", range(6))
def test_drain_matches_sorted(seed):
    """Build a random heap, drain it, and demand the exact sorted() order."""
    rng = random.Random(seed)
    pairs = [(bytes([i]), bytes([j])) for i in range(5) for j in range(4)]
    rng.shuffle(pairs)
    nodes = [node(p, rng.randint(1, 15)) for p in pairs]

    h = build(nodes)
    got = drain(h)

    assert as_keys(got) == as_keys(sorted(nodes, reverse=True))
    assert h.index == {}


@pytest.mark.parametrize("seed", range(6))
def test_push_update_drain_reconciles(seed):
    """
    Mirror of test_bpe.py's reconcile test: push, apply a random mix of
    updates while tracking ground-truth counts in a dict, then drain and
    assert the order matches sorting that dict.
    """
    rng = random.Random(seed)
    pairs = [(bytes([i]), bytes([j])) for i in range(4) for j in range(4)]
    counts = {p: rng.randint(1, 20) for p in pairs}

    h = IndexedHeap()
    for p in pairs:
        h.push(node(p, counts[p]))

    for _ in range(40):
        p = rng.choice(pairs)
        delta = rng.randint(-8, 8)
        h.update(node(p, 0), delta)
        counts[p] += delta
        assert_heap_property(h)
        assert_index_consistent(h)

    expected = sorted((node(p, c) for p, c in counts.items()), reverse=True)
    assert as_keys(drain(h)) == as_keys(expected)


def test_push_counts():
    node_1 = HeapNode(pair=(b'a', b'b'), count=3)
    node_2 = HeapNode(pair=(b'a', b'b'), count=99)
    
    hp = IndexedHeap()
    hp.push(node_1)
    assert len(hp.heap) == 1
    assert hp.index[node_1.pair] == 0

    hp.push(node_2)

    print(hp.heap[hp.index[node_2.pair]].count)