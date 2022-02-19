from find_my_bike.dataset.add_aspect import add_aspect

# noinspection PyUnresolvedReferences
from .assets import dummy_meta


def test_add_aspect(dummy_meta):
    changed_meta = add_aspect("test", dummy_meta)
    for image_file, image_info in changed_meta.items():
        assert "test" in image_info["labels"]
        assert image_info["labels"]["test"] is None
