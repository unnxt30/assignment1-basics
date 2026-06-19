from dataclasses import dataclass


@dataclass
class BPEOutput:
    vocabulary: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]


@dataclass
class MergePair:
    count: int
    indexes: set[int]