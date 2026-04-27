"""ALES NRM: Data reduction pipeline for LBTI/LMIRCam/ALES NRM observations.

This package provides tools for performing analysis of non-redundant
aperture masking (NRM) infrared imaging data obtained with the Arizona
Lenslets for Exoplanet Spectroscopy (ALES) instrument of the Large
Binocular Telescope Interferometer (LBTI).
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
