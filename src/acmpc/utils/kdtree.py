from scipy.spatial import KDTree as KDTreeBase


class KDTree(KDTreeBase):
    """
    Adds some list-like properties
    """

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self) -> int:
        return self.data.shape[0]
