"""DOWN model wrapper."""

from __future__ import annotations

from cryptomvp.models.baseline import BaselineMLP


class DownModel(BaselineMLP):
    """Model_2_DOWN with independent weights."""