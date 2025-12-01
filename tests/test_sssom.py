import pytest
from typing import List
from pathlib import Path

from ai_harmonization.harmonization_approaches.base import (
    SingleHarmonizationSuggestion,
    harmonization_suggestions_to_sssom,
)


@pytest.fixture
def example_suggestions():
    return [
        SingleHarmonizationSuggestion(
            source_node="Table1",
            source_property="foo",
            source_description="Foo field",
            target_node="TableA",
            target_property="bar",
            target_description="Bar field",
            similarity=0.88,
        ),
        SingleHarmonizationSuggestion(
            source_node="Table2",
            source_property="baz",
            source_description="Baz field",
            target_node="TableB",
            target_property="buzz",
            target_description="Buzz field",
            similarity=0.92,
        ),
    ]


def test_end_to_end_sssom(
    tmp_path: Path, example_suggestions: List[SingleHarmonizationSuggestion]
):
    """
    Takes a list of single harmonization suggestions and writes them to an output file in
    SSSOM format. The function then asserts that the output file exists,
    contains the expected content, and has the lines
    containing the expected mappings.
    """
    outpath = tmp_path / "output.sssom.tsv"

    harmonization_suggestions_to_sssom(example_suggestions, str(outpath))

    assert outpath.exists()
    content = outpath.read_text()

    assert "Table1_foo" in content
    assert "TableA_bar" in content
    assert "Foo field" in content
    assert "Buzz field" in content

    lines = [l for l in content.split("\n") if l and not l.startswith("#")]
    assert any("Table1_foo" in l and "TableA_bar" in l for l in lines)
    assert any("Table2_baz" in l and "TableB_buzz" in l for l in lines)
