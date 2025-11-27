from dataclasses import dataclass, field
from typing import Any, Dict, List
import numpy as np
from utils.io.plotutils import write_image

@dataclass
class IntermediateImage:
    name: str          
    image: np.ndarray  

@dataclass
class AnalysisResult:
    pipeline_name: str
    detections: List[Any]
    intermediates: List[IntermediateImage] | None = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_detections(self) -> bool:
        return len(self.detections) > 0
    
    def save(self, save_path: str) -> None:
        pass