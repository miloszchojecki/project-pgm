import os
from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class Config:
    project_root: str = "/home/soltys/sztuczna_inteligencja/sem1/pgm/project-pgm"
    image_size: int = 256
    batch_size: int = 64
    num_classes: int = 2
    epochs: int = 2
    learning_rate: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    @property
    def dataset_paths(self):
        return {
            "real_train": Path(self.project_root) / "dataset" / "train",
            "val": Path(self.project_root) / "dataset" / "test"
        }
    
    def get_gen_path(self, model_type: str) -> str:
        return Path(self.project_root) / "results" / model_type / "samples"
    
    def get_model_path(self, model_type: str) -> str:
        dir_name = "cnn" if model_type == "base" else model_type
        return Path(self.project_root) / "results" / dir_name / f"ClassicCNN_{model_type}.pth"
