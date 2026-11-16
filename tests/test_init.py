"""Tests for package-level metadata in ``ales_nrm/__init__.py``."""

import importlib
import sys
from importlib.metadata import version as dist_version

import pytest

import ales_nrm

_FALLBACK_VERSION = "0.0.0.dev0"


@pytest.fixture
def missing_version_module():
    """Make ``ales_nrm._version`` unimportable for one test.

    The patch is undone before ``ales_nrm`` is reloaded, so the real
    version is restored for all subsequent tests, even if the test
    fails.
    """
    with pytest.MonkeyPatch.context() as mp:
        # A None entry in sys.modules makes the import raise
        # ModuleNotFoundError (a subclass of ImportError).
        mp.setitem(sys.modules, "ales_nrm._version", None)
        yield
    importlib.reload(ales_nrm)


class TestVersion:
    """Tests for ``ales_nrm.__version__``."""

    def test_version_matches_distribution_metadata(self):
        """Installed version agrees with the package metadata."""
        assert ales_nrm.__version__ == dist_version("ales-nrm")

    def test_version_is_not_fallback_when_installed(self):
        """The fallback is not used when ``_version.py`` exists."""
        assert ales_nrm.__version__ != _FALLBACK_VERSION

    def test_fallback_when_version_file_missing(self, missing_version_module):
        """Use fallback version if ``_version`` cannot be imported."""
        reloaded = importlib.reload(ales_nrm)
        assert reloaded.__version__ == _FALLBACK_VERSION

    def test_all_exports_version(self):
        """``__all__`` exposes ``__version__``."""
        assert ales_nrm.__all__ == ["__version__"]
