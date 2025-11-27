from dataclasses import dataclass, field
from typing import Any, Dict, List
import numpy as np
from utils.io.plotutils import write_image

@dataclass
class IntermediateImage:
    name: str          
    image: np.ndarray
    
@dataclass
class SnowflakeDetection:
    label: int
    area: float
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    equivalent_diameter_area: float
    image_filled: np.ndarray

@dataclass
class AnalysisResult:
    pipeline_name: str
    detections: List[Any]
    labelled_image: np.ndarray
    intermediates: List[IntermediateImage]
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_detections(self) -> bool:
        return len(self.detections) > 0
    
    def save(self, save_path: str) -> None:
        """Saves the labelled image and intermediate images to the specified path."""
        write_image(self.labelled_image.astype(np.uint8)*255, save_path=save_path, filename="labelled_image.png")
        for intermediate in self.intermediates:
            write_image(intermediate.image.astype(np.uint8), save_path=save_path, filename=f"{intermediate.name}.png")