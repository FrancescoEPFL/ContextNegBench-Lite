from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ObjectSpec:
    shape: str
    color: str
    x: int
    y: int
    size: int
    has_border: bool = False
    border_color: str | None = None


@dataclass(frozen=True)
class Annotation:
    image_id: str
    image_path: str
    task_type: str
    objects: list[ObjectSpec]
    attributes: dict[str, Any]
    relation: str | None
    correct_caption: str
    hard_negative_captions: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["objects"] = [asdict(obj) for obj in self.objects]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Annotation":
        objects = [ObjectSpec(**obj) for obj in data.get("objects", [])]
        return cls(
            image_id=data["image_id"],
            image_path=data["image_path"],
            task_type=data["task_type"],
            objects=objects,
            attributes=data.get("attributes", {}),
            relation=data.get("relation"),
            correct_caption=data["correct_caption"],
            hard_negative_captions=list(data.get("hard_negative_captions", [])),
            metadata=data.get("metadata", {}),
            seed=int(data.get("seed", 0)),
        )
