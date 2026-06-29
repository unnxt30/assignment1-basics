from dataclasses import dataclass


@dataclass
class BPEOutput:
    vocabulary: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]


@dataclass
class MergePair:
    count: int
    indexes: set[int]


@dataclass
class HeapNode:
    count: int
    pair: tuple[bytes, bytes]

    def __lt__(self, other):
        if  other.count == self.count:
            return self.pair < other.pair
        return self.count < other.count
        
class IndexedHeap:
    def __init__(self):
        # parent = floor(i-1/2)
        self.heap: list[HeapNode] = []
        self.index: dict[tuple[bytes, bytes], int] = {}

    def push(self, node: HeapNode):
        self.heap.append(node)
        self.index[node.pair] = len(self.heap) - 1
        self.sift_up(len(self.heap)-1)
        return


    def delete(self, node:HeapNode):
        if len(self.heap) < 1:
            return 
        ind = self.index[node.pair]
        last = len(self.heap) - 1
        self.swap(ind, len(self.heap) - 1)
        del self.index[node.pair]
        self.heap = self.heap[:len(self.heap) - 1]
        if ind == last:
            return
        if ind == 0:
            self.sift_down(ind)
        else:
            parent = self._parent(ind)
            if parent != -1:
                if self.heap[parent] < self.heap[ind]:
                    self.sift_up(ind)
                else:
                    self.sift_down(ind)


    def pop_max(self):
        return self.heap[0]


    def update(self, node: HeapNode, delta: int):
        curr_ind = self.index.get(node.pair, -1)
        curr_count = self.heap[curr_ind].count
        self.heap[curr_ind] = HeapNode(pair=node.pair, count=curr_count + delta)
        parent = self._parent(curr_ind) 

        if parent != -1:
            if self.heap[parent] < self.heap[curr_ind]:
                self.sift_up(curr_ind)
            else:
                self.sift_down(curr_ind)
        else:
            # tacke the case when the root is counted down.
            if delta < 0:
                self.sift_down(curr_ind)

        return

    def sift_up(self, ind:int):
        parent = self._parent(ind)
        if parent == -1:
            return

        par = self.heap[parent]
        curr = self.heap[ind]

        while par < curr:
            if parent == -1:
                break

            self.swap(ind, parent)
            ind = parent
            parent = self._parent(parent)

            par = self.heap[parent]
            curr = self.heap[ind]


    def sift_down(self, ind:int):
        curr = ind
        swap = self.down_condition(ind)
        while swap != -1:
            self.swap(curr, swap)
            curr = swap
            swap = self.down_condition(curr)
    

    def swap(self, i:int, j:int):
        og = self.heap[i].pair
        swp = self.heap[j].pair
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        self.index[og] = j
        self.index[swp] = i

    def _parent(self, i) -> int :
        if i == 0:
            return -1
        return (i-1)//2

    def _right(self, i) -> int:
        ind = (2 * i) + 2
        if ind >= len(self.heap):
            return -1
        return ind

    def _left(self, i) -> int:
        ind = (2 * i) + 1
        if ind >= len(self.heap):
            return -1
        return ind
    
    def down_condition(self, i):
        if self._left(i) < 0:
            return -1

        if self._right(i) != -1:
            if self.heap[i] < self.heap[self._left(i)] and self.heap[i] < self.heap[self._right(i)]:
                return self._left(i) if self.heap[self._left(i)] > self.heap[self._right(i)] else self._right(i)
            elif self.heap[i] < self.heap[self._right(i)]:
                return self._right(i)
        if self.heap[i] < self.heap[self._left(i)]:
                return self._left(i)


        return -1
    