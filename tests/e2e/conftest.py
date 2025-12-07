"""End-to-end test fixtures - full system workflows."""
import pytest
from pathlib import Path


@pytest.fixture
def e2e_results_dir(temp_dir: Path) -> Path:
    """Temporary results directory for e2e test outputs."""
    results = temp_dir / "e2e_results"
    results.mkdir(parents=True)
    return results
