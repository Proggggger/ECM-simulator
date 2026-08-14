"""Tree-based pseudo-LRU replacement."""
from .base import ReplacementPolicyBase


class PseudoLRUReplacement(ReplacementPolicyBase):
    """Binary-tree pseudo-LRU: LRU's behaviour at a fraction of the state.

    True LRU needs an ordering over all ``assoc`` ways (log2(assoc!) bits per
    set); tree-PLRU needs only ``assoc - 1`` bits. Each internal node bit
    points at the half that was *least* recently used, so the victim is found
    by walking the bits from the root, and a touch flips the bits along its
    path to point AWAY from itself.

    This is what real L1s actually implement, and it makes a nice UI
    comparison: near-LRU hit rates, far less hardware. Requires ``assoc`` to
    be a power of two (real tree-PLRU does too); the cache validates that.
    """
    name = "plru"
    label = "Pseudo-LRU"
    description = "Tree-based approximation of LRU (assoc-1 bits per set)"

    def __init__(self, num_sets, assoc):
        if assoc & (assoc - 1):
            raise ValueError(
                f"tree-PLRU requires a power-of-two associativity, got {assoc}")
        super().__init__(num_sets, assoc)

    def reset(self):
        # assoc-1 internal nodes; 0 = "left subtree is the LRU half".
        self._bits = [[0] * (self.assoc - 1) for _ in range(self.num_sets)]

    def touch(self, set_idx, way):
        if self.assoc == 1:
            return
        bits = self._bits[set_idx]
        node = 0
        # Walk from the root to the leaf, flipping each bit to point at the
        # OTHER half (i.e. away from the way just accessed).
        for level in range(self.assoc.bit_length() - 1):
            # Which half of the remaining range does `way` live in?
            half = self.assoc >> (level + 1)
            went_right = bool(way & half)
            bits[node] = 0 if went_right else 1
            node = node * 2 + (2 if went_right else 1)
            if node >= len(bits):
                break

    def victim(self, set_idx):
        if self.assoc == 1:
            return 0
        bits = self._bits[set_idx]
        node, way = 0, 0
        for level in range(self.assoc.bit_length() - 1):
            go_right = bool(bits[node])
            half = self.assoc >> (level + 1)
            if go_right:
                way |= half
            node = node * 2 + (2 if go_right else 1)
            if node >= len(bits):
                break
        return way
