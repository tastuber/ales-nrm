"""NRM mask geometry and Fourier-plane coordinate generation.

This module loads pupil-plane mask hole coordinates,
computes baselines, generates synthetic power spectra, and
creates the mapping between baselines and splodge positions
in the Fourier plane. The approach follows SAMpy
(Sallum & Eisner 2017; Sallum et al. 2022).
"""

import logging
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np

from ales_nrm.utilities import ensure_odd

logger = logging.getLogger(__name__)

# ALES pixel scale in arcseconds per pixel.
ALES_PIXEL_SCALE_ARCSEC = 0.0345


@dataclass
class Hole:
    """A single hole in the NRM mask.

    Attributes:
        name: Hole identifier (e.g., ``'H1'``).
        x: X position in the pupil plane in meters.
        y: Y position in the pupil plane in meters.
        radius: Hole radius in meters.
    """

    name: str
    x: float
    y: float
    radius: float


@dataclass
class Baseline:
    """A baseline between two mask holes.

    Attributes:
        name: Baseline identifier (e.g., ``'H1H3'``).
        hole1: First hole name.
        hole2: Second hole name.
        bx: Baseline x-component in meters.
        by: Baseline y-component in meters.
        length: Baseline length in meters.
    """

    name: str
    hole1: str
    hole2: str
    bx: float
    by: float
    length: float


@dataclass
class NRMMask:
    """Non-redundant aperture mask geometry and Fourier mapping.

    Loads mask hole coordinates from a file, computes all
    unique baselines, and provides methods to generate
    synthetic power spectra and splodge position maps.

    Attributes:
        holes: List of Hole objects.
        baselines: List of Baseline objects.
        primary_diameter: Primary mirror diameter in
            meters.
    """

    primary_diameter: float
    holes: list[Hole] = field(default_factory=list)
    baselines: list[Baseline] = field(default_factory=list)

    @property
    def n_holes(self) -> int:
        """Number of holes in the mask."""
        return len(self.holes)

    @property
    def n_baselines(self) -> int:
        """Number of unique baselines."""
        return len(self.baselines)

    @property
    def n_closing_triangles(self) -> int:
        """Number of closing triangles."""
        n = self.n_holes
        return n * (n - 1) * (n - 2) // 6

    @classmethod
    def from_file(
        cls,
        filepath: str | Path,
        primary_diameter: float = 8.4,
    ) -> "NRMMask":
        """Load a mask from a coordinate file.

        The file format is whitespace-delimited with
        columns: hole_name, x_m, y_m, radius_m.
        Lines starting with '#' are comments.

        Args:
            filepath: Path to the mask coordinate file.
            primary_diameter: Primary mirror diameter in
                meters. Default is 8.4 m (LBT).

        Returns:
            NRMMask instance with holes and baselines.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is invalid.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Mask file not found: {filepath}")

        holes = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 4:
                    raise ValueError(
                        f"Expected 4 columns, got {len(parts)}: '{line}'"
                    )
                name = parts[0]
                x = float(parts[1])
                y = float(parts[2])
                radius = float(parts[3])
                holes.append(Hole(name, x, y, radius))

        mask = cls(
            holes=holes,
            primary_diameter=primary_diameter,
        )
        mask._compute_baselines()

        logger.info(
            "Loaded mask with %d holes and %d baselines from %s.",
            mask.n_holes,
            mask.n_baselines,
            filepath.name,
        )
        return mask

    @classmethod
    def from_bundled(
        cls,
        name: str = "lbti_nrm6_sx",
        primary_diameter: float = 8.4,
    ) -> "NRMMask":
        """Load a bundled mask file by name.

        Args:
            name: Mask name without extension. Available:
                ``'lbti_nrm6_sx'``,
                ``'lbti_nrm6_dx'``.
            primary_diameter: Primary mirror diameter.

        Returns:
            NRMMask instance.
        """
        masks_pkg = resources.files("ales_nrm.masks")
        mask_file = masks_pkg / f"{name}.txt"

        with resources.as_file(mask_file) as path:
            return cls.from_file(path, primary_diameter)

    def _compute_baselines(self) -> None:
        """Compute all unique baselines from hole pairs."""
        self.baselines = []
        for i in range(self.n_holes):
            for j in range(i + 1, self.n_holes):
                h1 = self.holes[i]
                h2 = self.holes[j]
                bx = h2.x - h1.x
                by = h2.y - h1.y
                length = np.sqrt(bx**2 + by**2)
                name = f"{h1.name}{h2.name}"
                self.baselines.append(
                    Baseline(
                        name,
                        h1.name,
                        h2.name,
                        bx,
                        by,
                        length,
                    )
                )

    def get_closing_triangles(
        self,
    ) -> list[tuple[str, str, str]]:
        """Compute all closing triangles of baselines.

        Returns:
            List of tuples, each containing three
            baseline names forming a closing triangle.
        """
        triangles = []
        for i in range(self.n_holes):
            for j in range(i + 1, self.n_holes):
                for k in range(j + 1, self.n_holes):
                    bl_ij = f"{self.holes[i].name}{self.holes[j].name}"
                    bl_jk = f"{self.holes[j].name}{self.holes[k].name}"
                    bl_ik = f"{self.holes[i].name}{self.holes[k].name}"
                    triangles.append((bl_ij, bl_jk, bl_ik))
        return triangles

    def make_pupil_image(
        self,
        n_pixels: int = 1001,
    ) -> tuple[np.ndarray, float]:
        """Generate an odd-shaped pupil-plane image of the mask.

        Creates a binary image showing the circular holes
        on the primary mirror. The image size is forced to
        odd to ensure a unique center pixel.

        Args:
            n_pixels: Image size in pixels. Default is 1001.
                Forced to odd if even.

        Returns:
            Tuple of (image, pixel_scale) where image is
            a 2D array and pixel_scale is meters/pixel.
        """
        n_pixels = ensure_odd(n_pixels)

        hole_xs = [h.x for h in self.holes]
        hole_ys = [h.y for h in self.holes]
        max_radius = max(h.radius for h in self.holes)

        # Extent covers from outermost hole edges plus
        # one radius of padding on each side.
        x_min = min(hole_xs) - 2 * max_radius
        x_max = max(hole_xs) + 2 * max_radius
        y_min = min(hole_ys) - 2 * max_radius
        y_max = max(hole_ys) + 2 * max_radius
        extent = max(
            x_max - x_min,
            y_max - y_min,
            self.primary_diameter,
        )

        pixel_scale = extent / n_pixels
        image = np.zeros((n_pixels, n_pixels))
        center_x = np.mean(hole_xs)
        center_y = np.mean(hole_ys)

        y_coords, x_coords = np.mgrid[0:n_pixels, 0:n_pixels]
        x_m = (x_coords - n_pixels // 2) * pixel_scale + center_x
        y_m = (y_coords - n_pixels // 2) * pixel_scale + center_y

        for hole in self.holes:
            dist = np.sqrt((x_m - hole.x) ** 2 + (y_m - hole.y) ** 2)
            image[dist <= hole.radius] = 1.0

        return image, pixel_scale

    def make_pupil_image_baseline(
        self,
        baseline: Baseline,
        n_pixels: int = 1001,
    ) -> tuple[np.ndarray, float]:
        """Generate a pupil image with only two holes.

        Used to identify which splodge belongs to which
        baseline by computing a two-hole power spectrum.

        Args:
            baseline: Baseline object specifying the two
                holes.
            n_pixels: Image size in pixels. Default is 1001.
                Forced to odd if even.

        Returns:
            Tuple of (image, pixel_scale).
        """
        n_pixels = ensure_odd(n_pixels)

        hole_xs = [h.x for h in self.holes]
        hole_ys = [h.y for h in self.holes]
        max_radius = max(h.radius for h in self.holes)

        x_min = min(hole_xs) - 2 * max_radius
        x_max = max(hole_xs) + 2 * max_radius
        y_min = min(hole_ys) - 2 * max_radius
        y_max = max(hole_ys) + 2 * max_radius
        extent = max(
            x_max - x_min,
            y_max - y_min,
            self.primary_diameter,
        )

        pixel_scale = extent / n_pixels
        image = np.zeros((n_pixels, n_pixels))
        center_x = np.mean(hole_xs)
        center_y = np.mean(hole_ys)

        y_coords, x_coords = np.mgrid[0:n_pixels, 0:n_pixels]
        x_m = (x_coords - n_pixels // 2) * pixel_scale + center_x
        y_m = (y_coords - n_pixels // 2) * pixel_scale + center_y

        for hole in self.holes:
            if hole.name in (
                baseline.hole1,
                baseline.hole2,
            ):
                dist = np.sqrt((x_m - hole.x) ** 2 + (y_m - hole.y) ** 2)
                image[dist <= hole.radius] = 1.0

        return image, pixel_scale

    def compute_synthetic_psf(
        self,
        wavelength: float,
        pixel_scale_arcsec: float = ALES_PIXEL_SCALE_ARCSEC,
        n_pixels_image: int = 101,
        n_pixels_pupil: int = 1001,
    ) -> np.ndarray:
        """Compute a monochromatic PSF from the mask.

        The output pixel scale is determined by the
        relationship between the FFT size and the pupil
        pixel scale:

            pixel_scale = lambda / (N_pad * pupil_pix_scale)

        where N_pad is the zero-padded array size. This
        method computes N_pad to achieve the requested
        pixel_scale_arcsec.

        Args:
            wavelength: Wavelength in microns.
            pixel_scale_arcsec: Image pixel scale in arcsec/pixel.
                Default is 0.0345 (ALES).
            n_pixels_image: Output image size in pixels. Default is 101.
                Forced to odd.
            n_pixels_pupil: Pupil sampling resolution. Default is 1001.
                Forced to odd.

        Returns:
            2D PSF image normalized to peak of 1.
        """
        n_pixels_image = ensure_odd(n_pixels_image)
        n_pixels_pupil = ensure_odd(n_pixels_pupil)

        pupil, pupil_pix_scale = self.make_pupil_image(n_pixels_pupil)

        wave_m = wavelength * 1e-6
        pixel_scale_rad = pixel_scale_arcsec * np.pi / (3600 * 180)

        # Padding to achieve desired image pixel scale.
        # Image pixel = lambda / (N * pupil_pixel_scale)
        n_pad = int(wave_m / (pixel_scale_rad * pupil_pix_scale))
        n_pad = max(n_pad, n_pixels_pupil, n_pixels_image)
        n_pad = ensure_odd(n_pad)

        padded = np.zeros((n_pad, n_pad))
        offset = (n_pad - n_pixels_pupil) // 2
        padded[
            offset : offset + n_pixels_pupil,
            offset : offset + n_pixels_pupil,
        ] = pupil

        efield = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(padded)))
        psf_full = np.abs(efield) ** 2

        # Crop centered output. With odd n_pad,
        # center is at n_pad // 2.
        cy = n_pad // 2
        cx = n_pad // 2
        half = n_pixels_image // 2
        psf = psf_full[
            cy - half : cy - half + n_pixels_image,
            cx - half : cx - half + n_pixels_image,
        ]

        peak = psf.max()
        if peak > 0:
            psf = psf / peak
        return psf

    def compute_synthetic_power_spectrum(
        self,
        wavelength: float,
        pixel_scale_arcsec: float = ALES_PIXEL_SCALE_ARCSEC,
        n_pixels_image: int = 101,
        n_pixels_pupil: int = 1001,
    ) -> np.ndarray:
        """Compute synthetic power spectrum showing splodges.

        This is |FFT(PSF)|^2, which shows the
        autocorrelation of the mask and reveals the
        splodge positions.

        Args:
            wavelength: Wavelength in microns.
            pixel_scale_arcsec: Image pixel scale in arcsec/pixel.
                Default is 0.0345 (ALES).
            n_pixels_image: Image size in pixels. Default is 101.
                Forced to odd.
            n_pixels_pupil: Pupil sampling resolution. Default is 1001.
                Forced to odd.

        Returns:
            2D power spectrum with DC at center.
        """
        psf = self.compute_synthetic_psf(
            wavelength,
            pixel_scale_arcsec,
            n_pixels_image,
            n_pixels_pupil,
        )
        ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(psf)))
        return np.abs(ft) ** 2

    def compute_splodge_positions(
        self,
        wavelengths: np.ndarray,
        pixel_scale_arcsec: float = ALES_PIXEL_SCALE_ARCSEC,
        ny: int = 67,
        nx: int = 67,
    ) -> dict[str, np.ndarray]:
        """Compute splodge center positions for all baselines.

        For each baseline, computes the (y, x) pixel
        coordinates in the centered FFT image where the
        baseline's signal appears, for each wavelength.

        Coordinates are in the centered FFT frame
        (zero frequency at center), matching
        ``np.fft.fftshift`` output and the
        ``ObservingBlock.power_spectra`` format.

        Args:
            wavelengths: 1D array of wavelengths in
                microns.
            pixel_scale_arcsec: Image pixel scale in
                arcsec/pixel. Default is 0.0345 (ALES).
            ny: Number of pixels in y dimension of the
                FFT image. Default is 67 (ALES padded).
            nx: Number of pixels in x dimension of the
                FFT image. Default is 67 (ALES padded).

        Returns:
            Dictionary mapping baseline names to arrays
            of shape ``(n_wavelengths, 2)`` containing
            ``(y_pixel, x_pixel)`` positions in the
            centered FFT frame.
        """
        pixel_scale_rad = pixel_scale_arcsec * np.pi / (3600 * 180)
        center_y = (ny - 1) / 2.0
        center_x = (nx - 1) / 2.0

        # Spatial frequency per pixel in each dimension.
        freq_per_pixel_x = 1.0 / (nx * pixel_scale_rad)
        freq_per_pixel_y = 1.0 / (ny * pixel_scale_rad)

        positions = {}
        for bl in self.baselines:
            coords = np.empty((len(wavelengths), 2))
            for i, wave_um in enumerate(wavelengths):
                wave_m = wave_um * 1e-6
                u = bl.bx / wave_m
                v = bl.by / wave_m
                px = u / freq_per_pixel_x
                py = v / freq_per_pixel_y
                coords[i] = [
                    center_y + py,
                    center_x + px,
                ]
            positions[bl.name] = coords

        logger.info(
            "Computed splodge positions for %d baselines "
            "at %d wavelengths (image %dx%d).",
            self.n_baselines,
            len(wavelengths),
            ny,
            nx,
        )
        return positions

    def plot_pupil(self, n_pixels: int = 1001, ax=None):
        """Plot the pupil-plane mask image.

        Args:
            n_pixels: Pupil image resolution.
            ax: Optional matplotlib axes.

        Returns:
            Matplotlib axes object.
        """
        import matplotlib.pyplot as plt

        image, pix_scale = self.make_pupil_image(n_pixels)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 6))

        hole_xs = [h.x for h in self.holes]
        hole_ys = [h.y for h in self.holes]
        center_x = np.mean(hole_xs)
        center_y = np.mean(hole_ys)
        n_actual = image.shape[0]
        half_extent = (n_actual * pix_scale) / 2.0

        ax.imshow(
            image,
            origin="lower",
            extent=[
                center_x - half_extent,
                center_x + half_extent,
                center_y - half_extent,
                center_y + half_extent,
            ],
            cmap="gray",
        )
        for hole in self.holes:
            ax.annotate(
                hole.name,
                (hole.x, hole.y),
                color="red",
                fontsize=8,
                ha="center",
                va="bottom",
            )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Pupil Plane Mask")
        ax.set_aspect("equal")
        return ax

    def plot_synthetic_psf(
        self,
        wavelength: float,
        pixel_scale_arcsec: float = ALES_PIXEL_SCALE_ARCSEC,
        n_pixels_image: int = 101,
        n_pixels_pupil: int = 1001,
        ax=None,
        log_scale: bool = True,
    ):
        """Plot the synthetic PSF.

        Args:
            wavelength: Wavelength in microns.
            pixel_scale_arcsec: Pixel scale in arcsec/pix.
                Default is 0.0345 (ALES).
            n_pixels_image: Image size. Default is 101.
            n_pixels_pupil: Pupil resolution. Default is 1001.
            ax: Optional matplotlib axes.
            log_scale: Use log stretch. Default is True.

        Returns:
            Matplotlib axes object.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        psf = self.compute_synthetic_psf(
            wavelength,
            pixel_scale_arcsec,
            n_pixels_image,
            n_pixels_pupil,
        )

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 6))

        norm = LogNorm(vmin=1e-4, vmax=1) if log_scale else None
        ax.imshow(psf, origin="lower", norm=norm)
        ax.set_title(f"Synthetic PSF ({wavelength:.2f} \u00b5m)")
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        return ax

    def plot_synthetic_power_spectrum(
        self,
        wavelength: float,
        pixel_scale_arcsec: float = ALES_PIXEL_SCALE_ARCSEC,
        n_pixels_image: int = 101,
        n_pixels_pupil: int = 1001,
        ax=None,
        log_scale: bool = True,
    ):
        """Plot the synthetic power spectrum.

        Args:
            wavelength: Wavelength in microns.
            pixel_scale_arcsec: Pixel scale in arcsec/pix.
                Default is 0.0345 (ALES).
            n_pixels_image: Image size. Default is 101.
            n_pixels_pupil: Pupil resolution. Default is 1001.
            ax: Optional matplotlib axes.
            log_scale: Use log stretch. Default is True.

        Returns:
            Matplotlib axes object.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        ps = self.compute_synthetic_power_spectrum(
            wavelength,
            pixel_scale_arcsec,
            n_pixels_image,
            n_pixels_pupil,
        )

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 6))

        norm = None
        if log_scale:
            ps_pos = ps.copy()
            ps_pos[ps_pos <= 0] = ps_pos[ps_pos > 0].min()
            norm = LogNorm(
                vmin=ps_pos.max() * 1e-6,
                vmax=ps_pos.max(),
            )

        ax.imshow(ps, origin="lower", norm=norm)
        ax.set_title(f"Synthetic Power Spectrum ({wavelength:.2f} \u00b5m)")
        ax.set_xlabel("u (pixels)")
        ax.set_ylabel("v (pixels)")
        return ax

    def plot_power_spectrum_with_baselines(
        self,
        power_spectrum: np.ndarray,
        wavelength: float,
        pixel_scale_arcsec: float = ALES_PIXEL_SCALE_ARCSEC,
        ax=None,
        log_scale: bool = True,
    ):
        """Plot measured power spectrum with splodge markers.

        Overlays computed baseline positions on the
        measured power spectrum for verification.

        Args:
            power_spectrum: 2D array from a single frame
                and wavelength (fftshift applied).
            wavelength: Wavelength in microns.
            pixel_scale_arcsec: Pixel scale in
                arcsec/pix. Default is 0.0345 (ALES).
            ax: Optional matplotlib axes.
            log_scale: Use log stretch. Default is True.

        Returns:
            Matplotlib axes object.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        ny, nx = power_spectrum.shape
        wavelengths = np.array([wavelength])
        positions = self.compute_splodge_positions(
            wavelengths,
            pixel_scale_arcsec,
            ny,
            nx,
        )

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 8))

        if log_scale:
            ps = power_spectrum.copy()
            ps[ps <= 0] = ps[ps > 0].min()
            norm = LogNorm(vmin=ps.max() * 1e-6, vmax=ps.max())
            ax.imshow(ps, origin="lower", norm=norm)
        else:
            ax.imshow(power_spectrum, origin="lower")

        center_y = (ny - 1) / 2.0
        center_x = (nx - 1) / 2.0

        for bl_name, coords in positions.items():
            y, x = coords[0]
            ax.plot(x, y, "r+", markersize=10)
            ax.annotate(
                bl_name,
                (x, y),
                color="red",
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
            # Conjugate splodge.
            y_conj = 2 * center_y - y
            x_conj = 2 * center_x - x
            ax.plot(x_conj, y_conj, "b+", markersize=10)

        ax.set_title(
            f"Power Spectrum with Baselines ({wavelength:.2f} \u00b5m)"
        )
        ax.set_xlabel("u (pixels)")
        ax.set_ylabel("v (pixels)")
        return ax
