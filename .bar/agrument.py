

import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class DataArguments: # 没必要在这里定义，@Jinhui TODO 移动到统一config 来定义。 
    #这里至少为了能够接受来着 terminal的 参数。 但是还是建议统一书写
    dataset_use: str = field(default="")
    eval_dataset: Optional[str] = field(default=None)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    
