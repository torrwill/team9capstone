"""Paper-subset 450/550 split with balanced speaker representation.

Refactored from notebook cell 21.
"""
from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def create_paper_split(npz_paths: list[Path],
                       speakers: list[str] | None = None,
                       samples_per_speaker: int = 200,
                       train_size: int = 450,
                       seed: int = 42) -> tuple[list[Path], list[Path]]:
    """Reproduce the paper's 1,000-sample / 450-550 split with balanced
    speaker representation. Per speaker: 90 train, 110 test (= 450/5, 550/5).
    """
    speaker_files: dict[str, list[Path]] = defaultdict(list)
    for p in npz_paths:
        sid = p.parent.name
        if speakers and sid not in speakers:
            continue
        speaker_files[sid].append(p)

    if speakers is None:
        speakers = sorted(speaker_files.keys(), key=lambda x: int(x[1:]))

    n_speakers = len(speakers)
    train_per_speaker = train_size // n_speakers
    test_per_speaker = samples_per_speaker - train_per_speaker

    rng = random.Random(seed)
    train_paths: list[Path] = []
    test_paths: list[Path] = []

    for sid in sorted(speakers, key=lambda x: int(x[1:])):
        files = sorted(speaker_files[sid])
        assert len(files) >= samples_per_speaker, \
            f"{sid} has {len(files)} files, need {samples_per_speaker}"

        sampled = rng.sample(files, samples_per_speaker)
        rng.shuffle(sampled)

        train_paths.extend(sampled[:train_per_speaker])
        test_paths.extend(sampled[train_per_speaker:])

    rng.shuffle(train_paths)
    rng.shuffle(test_paths)

    logger.info(
        "split: %d train / %d test across %d speakers (%d/%d per speaker)",
        len(train_paths), len(test_paths), n_speakers,
        train_per_speaker, test_per_speaker,
    )

    return train_paths, test_paths
