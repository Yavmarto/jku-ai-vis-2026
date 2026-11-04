"""Checks on the files the course and the group exports rely on. Each check skips if its file
does not exist yet, so CI stays green until the file is added and then guards it."""
import csv
import json
from pathlib import Path

import pytest

from lab.config import CSV_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


@pytest.mark.parametrize("name", ["team-info.json", "workload.json"])
def test_course_json_parses(name):
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} not added yet (course-provided)")
    json.loads(path.read_text(encoding="utf-8"))


def test_readme_exists():
    if not (ROOT / "README.md").exists():
        pytest.skip("README.md not added yet")


@pytest.mark.parametrize("name", sorted(CSV_COLUMNS))
def test_result_csv_columns(name):
    paths = list(RESULTS.glob(f"*/{name}"))  # results/<model name>/<file>
    if not paths:
        pytest.skip(f"no results/*/{name} exported yet")
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        assert header == CSV_COLUMNS[name], path
