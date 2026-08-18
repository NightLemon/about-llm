import pytest

from scripts.check_repo_hygiene import forbidden_generated_paths

pytestmark = pytest.mark.contract


def test_forbidden_generated_paths_normalizes_and_filters() -> None:
    assert forbidden_generated_paths(
        [
            "docs/index.md",
            ".review-site/index.html",
            ".windows-site\\assets\\main.css",
            "site",
            "website/source.md",
            "",
        ]
    ) == [
        ".review-site/index.html",
        ".windows-site/assets/main.css",
        "site",
    ]
