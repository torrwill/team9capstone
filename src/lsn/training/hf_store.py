"""HuggingFace Hub adapter — the ONLY module that imports huggingface_hub.

Spec §9.3: gated by config.checkpointing.hf_repo. When unset, the runner
passes None everywhere and zero HF code paths are exercised.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

logger = logging.getLogger(__name__)


class HFStore:
    """Thin wrapper around HfApi for upload/download of checkpoints.

    `token=None` reads the HF_TOKEN env var. The README documents how each
    environment (Colab/Kaggle/local) populates HF_TOKEN.
    """

    def __init__(self, repo: str, subfolder: str, token: str | None = None):
        self.repo = repo
        self.subfolder = subfolder
        self.token = token if token is not None else os.environ.get("HF_TOKEN")
        self._api: HfApi | None = None

    @property
    def api(self) -> HfApi:
        if self._api is None:
            self._api = HfApi(token=self.token)
        return self._api

    def upload(self, local_path: Path, remote_filename: str, *,
               commit_message: str) -> bool:
        """Upload `local_path` to `<subfolder>/<remote_filename>`. Returns True
        on success; logs a warning and returns False on failure (never raises —
        HF hiccups must not kill training)."""
        try:
            self.api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=f"{self.subfolder}/{remote_filename}",
                repo_id=self.repo,
                repo_type="model",
                commit_message=commit_message,
            )
            return True
        except Exception as e:
            logger.warning("HF upload failed: %s", e)
            return False

    def try_download(self, remote_filename: str, local_dest: Path) -> Path | None:
        """Download `<subfolder>/<remote_filename>` into `local_dest` directory.
        Returns the local Path on success; None if the file isn't on the Hub
        (clean miss, not an error)."""
        try:
            local_dest.mkdir(parents=True, exist_ok=True)
            path = hf_hub_download(
                repo_id=self.repo,
                filename=f"{self.subfolder}/{remote_filename}",
                token=self.token,
                local_dir=str(local_dest),
            )
            return Path(path)
        except (EntryNotFoundError, RepositoryNotFoundError):
            return None
        except Exception as e:
            logger.info("HF download error for %s: %s", remote_filename, e)
            return None
