from negcompbench.data.captions import attribute_binding_caption, object_phrase, prompt_variants, spatial_caption


def test_attribute_binding_caption_swaps_are_distinct():
    correct = attribute_binding_caption("red", "square", "blue", "circle")
    negative = attribute_binding_caption("blue", "square", "red", "circle")
    assert correct == "a red square and a blue circle"
    assert negative == "a blue square and a red circle"
    assert correct != negative


def test_spatial_caption_and_variants():
    caption = spatial_caption("red", "square", "left_of", "blue", "circle")
    variants = prompt_variants("red", "square", "left_of", "blue", "circle")
    assert "left of" in caption
    assert len(variants) == 3
    assert caption in variants


def test_count_caption_pluralizes():
    assert object_phrase("blue", "circle", 3) == "three blue circles"
