from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnglishObjectForm:
    text: str
    use_article: bool

    @property
    def noun_phrase(self) -> str:
        return f"a {self.text}" if self.use_article else self.text

    @property
    def visible_phrase(self) -> str:
        return f"a visible {self.text}" if self.use_article else f"visible {self.text}"


SCENARIO_OBJECT_FORMS = {
    ("street", "car"): EnglishObjectForm("cars", use_article=False),
}


def english_object_form(scene: str, object_name: str) -> EnglishObjectForm:
    scene_key = scene.strip().lower()
    object_key = object_name.strip().lower()
    return SCENARIO_OBJECT_FORMS.get((scene_key, object_key), EnglishObjectForm(object_key, use_article=True))
