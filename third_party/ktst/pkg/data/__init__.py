from pathlib import Path
from typing import Optional

import pandas as pd
from torch.utils.data import DataLoader


class BaseData:
    train_loader: DataLoader
    val_loader: DataLoader
    num_questions: int
    num_concepts: int
    max_concepts: int
    max_len: int
    format: str

    test_file: Path | pd.DataFrame
    test_loader: Optional[DataLoader]
    original_max_concepts: int | None
    unique_concept_mapping: dict[tuple, int] | None
    swap_q_and_c: bool
