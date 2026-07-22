"""Compatibility name for D4CMPP2's generalized ISA PyG dataset."""

from D4CMPP2.src.DataManager.Dataset.ISAGraphDataset import ISAGraphDataset


class ISAGraphDataset_withSolv(ISAGraphDataset):
    """Use the canonical multi-molecule ISA dataset without a DGL adapter."""
