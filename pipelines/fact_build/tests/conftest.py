from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "contracts" / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "pipelines" / "fact_build" / "src"))


@pytest.fixture(scope="session")
def fact_config():
    from fact_build.config import FactBuildConfig

    return FactBuildConfig.load(REPOSITORY_ROOT / "config" / "facts" / "build.json")


@pytest.fixture(scope="session")
def fact_builder(fact_config):
    from fact_build.builder import FactBuilder

    return FactBuilder(fact_config, local_files_only=True)
