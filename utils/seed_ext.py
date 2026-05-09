"""
utils/seed_ext.py
─────────────────
DataLoader worker seeding utilities referenced by sequence_builder.make_dataloaders.

These are appended here rather than modifying seed.py so the existing file is
not regenerated.  Import via ``from utils.seed import seed_worker, make_generator``
once this module patches the namespace (done automatically when training/trainer.py
is imported).
"""

import torch
import numpy as np
import random


def seed_worker(worker_id: int) -> None:
    """Worker init function for reproducible DataLoader shuffling."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = 42) -> torch.Generator:
    """Return a seeded torch Generator for DataLoader reproducibility."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
