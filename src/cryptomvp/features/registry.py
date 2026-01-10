"""Feature registry and feature set loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml


@dataclass(frozen=True)
class FeatureSpec:
    id: str
    category: str
    requires_extra_data: bool
    default_enabled: bool
    compute_cost: str
    params: Dict[str, object]
    description: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_registry_path() -> Path:
    return _repo_root() / "features" / "registry.yaml"


def default_feature_sets_path() -> Path:
    return _repo_root() / "features" / "feature_sets.yaml"


def load_registry(path: Optional[Path] = None) -> Dict[str, FeatureSpec]:
    """Load the feature registry."""
    path = path or default_registry_path()
    data = yaml.safe_load(path.read_text())
    specs: Dict[str, FeatureSpec] = {}
    for item in data.get("features", []):
        spec = FeatureSpec(
            id=str(item["id"]),
            category=str(item.get("category", "")),
            requires_extra_data=bool(item.get("requires_extra_data", False)),
            default_enabled=bool(item.get("default_enabled", False)),
            compute_cost=str(item.get("compute_cost", "cheap")),
            params=dict(item.get("params", {})),
            description=str(item.get("description", "")),
        )
        specs[spec.id] = spec
    return specs


def load_feature_sets(path: Optional[Path] = None) -> Dict[str, List[str]]:
    """Load feature set definitions."""
    path = path or default_feature_sets_path()
    data = yaml.safe_load(path.read_text())
    sets: Dict[str, List[str]] = {}
    for name, entry in data.get("feature_sets", {}).items():
        sets[name] = list(entry.get("features", []))
    return sets


def resolve_feature_list(
    list_of_features: Iterable[str],
    feature_set_id: Optional[str],
    feature_sets_path: Optional[Path] = None,
) -> List[str]:
    """Resolve the final feature list based on optional feature_set_id."""
    if feature_set_id is None:
        return list(list_of_features)
    feature_sets = load_feature_sets(feature_sets_path)
    if feature_set_id not in feature_sets:
        raise ValueError(f"Unknown feature_set_id: {feature_set_id}")
    return list(feature_sets[feature_set_id])


def validate_feature_ids(
    feature_ids: Iterable[str],
    registry: Optional[Dict[str, FeatureSpec]] = None,
    allow_extra_data: bool = False,
) -> List[str]:
    """Validate feature ids against registry."""
    registry = registry or load_registry()
    missing = [feat for feat in feature_ids if feat not in registry]
    if missing:
        raise ValueError(f"Unknown features: {missing}")
    if not allow_extra_data:
        blocked = [feat for feat in feature_ids if registry[feat].requires_extra_data]
        if blocked:
            raise ValueError(f"Features require extra data: {blocked}")
    return list(feature_ids)
