from __future__ import annotations

NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def article(word: str) -> str:
    return "an" if word[0].lower() in {"a", "e", "i", "o", "u"} else "a"


def object_phrase(color: str, shape: str, count: int | None = None) -> str:
    if count is None or count == 1:
        return f"{article(color)} {color} {shape}"
    plural = pluralize(shape)
    return f"{NUMBER_WORDS[count]} {color} {plural}"


def pluralize(shape: str) -> str:
    if shape.endswith("s"):
        return shape
    return f"{shape}s"


def attribute_binding_caption(color_a: str, shape_a: str, color_b: str, shape_b: str) -> str:
    return f"{article(color_a)} {color_a} {shape_a} and {article(color_b)} {color_b} {shape_b}"


def spatial_caption(color_a: str, shape_a: str, relation: str, color_b: str, shape_b: str) -> str:
    if relation == "left_of":
        phrase = "to the left of"
    elif relation == "right_of":
        phrase = "to the right of"
    elif relation == "above":
        phrase = "above"
    elif relation == "below":
        phrase = "below"
    else:
        raise ValueError(f"Unsupported relation: {relation}")
    return f"{article(color_a)} {color_a} {shape_a} {phrase} {article(color_b)} {color_b} {shape_b}"


def prompt_variants(color_a: str, shape_a: str, relation: str, color_b: str, shape_b: str) -> list[str]:
    base = spatial_caption(color_a, shape_a, relation, color_b, shape_b)
    if relation == "left_of":
        inverse = f"{article(color_b)} {color_b} {shape_b} to the right of {article(color_a)} {color_a} {shape_a}"
        natural = f"there is {article(color_a)} {color_a} {shape_a} on the left side of {article(color_b)} {color_b} {shape_b}"
    elif relation == "above":
        inverse = f"{article(color_b)} {color_b} {shape_b} below {article(color_a)} {color_a} {shape_a}"
        natural = f"there is {article(color_a)} {color_a} {shape_a} above {article(color_b)} {color_b} {shape_b}"
    else:
        inverse = base
        natural = base
    return [base, inverse, natural]
