import torch
import os
import numpy as np

from abc import ABC, abstractmethod
from pocaduck import Query, StorageConfig

class DataInterface(ABC):
    def __init__(self, pth: str):
        self.pth = pth

    @abstractmethod
    def load_pc(self, root_id: int):
        pass

class PoCADuckInterface(DataInterface):
    def __init__(self, pth: str):
        super().__init__(pth)

    def load_pc(self, root_id: int) -> tuple[torch.tensor, torch.tensor]:
        config = StorageConfig(base_path=self.pth)
        query = Query(config)
        data = query.get_points(label=root_id)
        query.close()

        points = data[..., :3]
        sv_ids = data[..., 3]

        points = torch.tensor(points, dtype=torch.float32)
        sv_ids = torch.tensor(sv_ids, dtype=torch.int64)

        return points, sv_ids

class DebugInterface(DataInterface):
    def __init__(self, pth: str):
        super().__init__(pth)

    def load_pc(self, root_id: int) -> tuple[torch.tensor, torch.tensor]:
        path = os.path.join(self.pth, f"label_{root_id}")
        files = os.listdir(path)
        data = np.concatenate([np.load(os.path.join(path, f))["points"] for f in files])

        points = data[..., :3]
        sv_ids = data[..., 3]

        points = torch.tensor(points)
        sv_ids = torch.tensor(sv_ids)

        return points, sv_ids