"""
Load pre-extracted PyRadiomics CSV rows for fusion with CNN features.

The CSV is produced by batch.extract_train_val_split_features (native NIfTI,
tumor ROI). Training only looks the vectors up — extraction stays offline so
the Trainer script does not change.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Columns that identify a row; everything else is a numeric feature.
_META_COLUMNS = frozenset({"case_id", "roi", "split"})


class RadiomicsFeatureTable:
    """
    Per-case radiomics vectors with train-only z-score normalisation.

    Usage:
        table = RadiomicsFeatureTable.from_csv(path)
        table.fit_normalize(train_case_ids)
        x = table.vector(case_id)   # float32 (n_features,)
    """

    def __init__(
        self,
        features: Dict[str, np.ndarray],
        feature_names: Sequence[str],
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        self.feature_names: List[str] = list(feature_names)
        self._raw: Dict[str, np.ndarray] = {
            case_id: np.asarray(vector, dtype=np.float32).reshape(-1)
            for case_id, vector in features.items()
        }
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        if mean is not None and std is not None:
            self.mean = np.asarray(mean, dtype=np.float32).reshape(-1)
            self.std = np.asarray(std, dtype=np.float32).reshape(-1)

        expected = len(self.feature_names)
        for case_id, vector in self._raw.items():
            if vector.shape[0] != expected:
                raise ValueError(
                    f"{case_id}: expected {expected} features, got {vector.shape[0]}"
                )

    @classmethod
    def from_csv(cls, path: Path) -> "RadiomicsFeatureTable":
        """
        Read a PyRadiomics batch CSV (case_id + numeric columns).

        Args:
            path: CSV written by save_features_csv / extract_*_features.
        """
        import csv

        path = Path(path)
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Empty radiomics CSV: {path}")
            if "case_id" not in reader.fieldnames:
                raise ValueError(f"Radiomics CSV missing case_id column: {path}")

            feature_names = [c for c in reader.fieldnames if c not in _META_COLUMNS]
            features: Dict[str, np.ndarray] = {}
            for row in reader:
                case_id = row["case_id"]
                values = np.empty(len(feature_names), dtype=np.float32)
                for i, name in enumerate(feature_names):
                    raw = row.get(name, "")
                    values[i] = 0.0 if raw in ("", None) else float(raw)
                # Prefer the first row if a case appears twice.
                features.setdefault(case_id, values)

        if not features:
            raise ValueError(f"No radiomics rows in {path}")

        return cls(features=features, feature_names=feature_names)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def case_ids(self) -> List[str]:
        return sorted(self._raw)

    def __contains__(self, case_id: str) -> bool:
        return case_id in self._raw

    def __len__(self) -> int:
        return len(self._raw)

    def fit_normalize(self, train_case_ids: Iterable[str]) -> "RadiomicsFeatureTable":
        """
        Fit mean/std on the training cases only (no val leakage).

        Missing train IDs are skipped. Columns with near-zero std become 1 so
        division stays safe; NaNs in the raw matrix are treated as 0 before fit.
        """
        rows = []
        for case_id in train_case_ids:
            if case_id not in self._raw:
                continue
            vector = np.nan_to_num(self._raw[case_id], nan=0.0, posinf=0.0, neginf=0.0)
            rows.append(vector)

        if not rows:
            raise ValueError("No overlapping train case IDs to fit radiomics normalisation.")

        stacked = np.stack(rows, axis=0)
        mean = stacked.mean(axis=0).astype(np.float32)
        std = stacked.std(axis=0).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

        self.mean = mean
        self.std = std
        return self

    def vector(self, case_id: str) -> np.ndarray:
        """
        Return a float32 feature vector for case_id.

        Missing cases become zeros (same length) so the batch stays rectangular.
        When fit_normalize has run, the vector is z-scored.
        """
        if case_id in self._raw:
            raw = np.nan_to_num(self._raw[case_id], nan=0.0, posinf=0.0, neginf=0.0)
        else:
            raw = np.zeros(self.n_features, dtype=np.float32)

        if self.mean is None or self.std is None:
            return raw.astype(np.float32, copy=False)

        return ((raw - self.mean) / self.std).astype(np.float32)

    def save_stats(self, path: Path) -> None:
        """Persist feature names + train mean/std next to checkpoints."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Call fit_normalize before save_stats.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_names": self.feature_names,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "n_features": self.n_features,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_stats(self, path: Path) -> "RadiomicsFeatureTable":
        """Apply previously fitted mean/std (must match feature_names order)."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        names = payload.get("feature_names")
        if names is not None and list(names) != self.feature_names:
            raise ValueError(
                "Radiomics stats feature_names do not match the CSV columns."
            )
        self.mean = np.asarray(payload["mean"], dtype=np.float32)
        self.std = np.asarray(payload["std"], dtype=np.float32)
        return self


def infer_radiomics_dim(dataset) -> int:
    """
    Radiomics vector width attached to a Dataset / Subset, else 0.

    Lets the notebook size MultiTaskUNet3D(radiomics_dim=...) from the same
    loader the Trainer will see, without changing Trainer(...).train(...).
    """
    from torch.utils.data import Subset

    current = dataset
    while isinstance(current, Subset):
        current = current.dataset

    table = getattr(current, "radiomics", None)
    if table is None:
        return 0
    return int(table.n_features)
