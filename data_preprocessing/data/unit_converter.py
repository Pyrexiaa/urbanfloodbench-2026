"""Optional unit conversion for the 1D-2D urban flood benchmark.

Different source models are built in different measurement systems.
Models 1-3 use **US customary** units (feet, cfs, inches of rain,
etc.) while Model 4 uses **SI** units (metres, cms, mm of rain).

This module provides an *optional* :class:`UnitConverter` that rescales the
assembled feature arrays from a model's native unit system into a common
target system (SI by default). It is deliberately decoupled from feature
extraction and normalization: it operates on the same ``(feature_list,
feature_vector)`` interface used by :class:`DatasetNormalizer`, and can be
applied as a single post-extraction step. Conversion is **off by default** and
only runs when a caller explicitly requests a target system that differs from
the model's native one.

"""

from __future__ import annotations

import numpy as np
from numpy import ndarray
from typing import Dict, List, Literal, Optional

UnitSystem = Literal["SI", "US"]

# --------------------------------------------------------------------------- #
# Physical dimensions                                                         #
# --------------------------------------------------------------------------- #
# Each dimension has a single multiplicative factor that converts a value from
# US customary units to SI units. The reverse (SI -> US) is 1 / factor.
LENGTH = "length"  # ft  -> m
AREA = "area"  # ft2 -> m2
VOLUME = "volume"  # ft3 -> m3
FLOW = "flow"  # ft3/s (cfs) -> m3/s (cms)
VELOCITY = "velocity"  # ft/s -> m/s
PRECIP_DEPTH = "precip_depth"  # in  -> mm
INVERSE_LENGTH = "inverse_length"  # 1/ft -> 1/m  (e.g. terrain curvature)
DIMENSIONLESS = "dimensionless"  # never scaled

# Base linear conversion: 1 international foot == 0.3048 m (exact).
_FT_TO_M = 0.3048
# 1 inch == 25.4 mm (exact).
_IN_TO_MM = 25.4

# US-customary -> SI multiplicative factors, keyed by dimension.
US_TO_SI_FACTORS: Dict[str, float] = {
    LENGTH: _FT_TO_M,
    AREA: _FT_TO_M**2,
    VOLUME: _FT_TO_M**3,
    FLOW: _FT_TO_M**3,  # cfs -> cms shares the volume factor
    VELOCITY: _FT_TO_M,
    PRECIP_DEPTH: _IN_TO_MM,
    INVERSE_LENGTH: 1.0 / _FT_TO_M,
    DIMENSIONLESS: 1.0,
}

# --------------------------------------------------------------------------- #
# Feature -> dimension map                                                    #
# --------------------------------------------------------------------------- #
# Feature names are consistent across the 1D/2D and node/edge groups, so a
# single global map is sufficient. Any feature *not* listed here is treated as
# DIMENSIONLESS (i.e. left unchanged) and a warning is emitted so new features
# are not silently mishandled.
FEATURE_DIMENSIONS: Dict[str, str] = {
    # --- positions / geometry (lengths) ---
    "position_x": LENGTH,
    "position_y": LENGTH,
    "relative_position_x": LENGTH,
    "relative_position_y": LENGTH,
    "length": LENGTH,
    "face_length": LENGTH,
    "diameter": LENGTH,
    # --- elevations / depths / stages (lengths) ---
    "elevation": LENGTH,
    "min_elevation": LENGTH,
    "invert_elevation": LENGTH,
    "surface_elevation": LENGTH,
    "depth": LENGTH,
    "water_level": LENGTH,
    "water_depth": LENGTH,
    # --- areas ---
    "area": AREA,
    "base_area": AREA,
    # --- volumes ---
    "water_volume": VOLUME,
    # --- flows (volumetric rate) ---
    "flow": FLOW,
    "inflow": FLOW,
    "inlet_flow": FLOW,
    # --- velocities ---
    "velocity": VELOCITY,
    # --- rainfall depth (inches vs mm, NOT feet vs metres) ---
    "rainfall": PRECIP_DEPTH,
    # --- terrain curvature (1 / horizontal length) ---
    "curvature": INVERSE_LENGTH,
    # --- unit-independent quantities (never scaled) ---
    "roughness": DIMENSIONLESS,  # Manning's n
    "slope": DIMENSIONLESS,  # rise / run
    "shape": DIMENSIONLESS,  # categorical conduit-shape code
    "aspect": DIMENSIONLESS,  # terrain aspect in degrees
    "flow_accumulation": DIMENSIONLESS,  # upstream cell count
    "direction_x": DIMENSIONLESS,  # unit normal-vector component
    "direction_y": DIMENSIONLESS,  # unit normal-vector component
}


class UnitConverter:
    """Rescales assembled feature arrays between US customary and SI units.

    Parameters
    ----------
    source_system:
        The native unit system of the model whose features are being
        converted (``"US"`` or ``"SI"``).
    target_system:
        The desired output unit system (``"US"`` or ``"SI"``).
    log_func:
        Callable used for informational/warning messages (defaults to ``print``).
    strict:
        If ``True``, an unknown feature name raises ``KeyError`` instead of
        being treated as dimensionless.
    """

    def __init__(
        self,
        source_system: UnitSystem,
        target_system: UnitSystem,
        log_func=print,
        strict: bool = False,
    ):
        source_system = source_system.upper()
        target_system = target_system.upper()
        for name, value in (
            ("source_system", source_system),
            ("target_system", target_system),
        ):
            if value not in ("SI", "US"):
                raise ValueError(f"{name} must be 'SI' or 'US', got {value!r}.")

        self.source_system = source_system
        self.target_system = target_system
        self.log_func = log_func
        self.strict = strict

    # ------------------------------------------------------------------ #
    # Introspection                                                       #
    # ------------------------------------------------------------------ #
    @property
    def is_noop(self) -> bool:
        """True when source and target systems are identical (nothing to do)."""
        return self.source_system == self.target_system

    def get_factor(self, feature: str) -> float:
        """Return the multiplicative factor to convert one ``feature`` value
        from ``source_system`` to ``target_system``.

        Unknown features return ``1.0`` (or raise if ``strict``).
        """
        if self.is_noop:
            return 1.0

        if feature not in FEATURE_DIMENSIONS:
            msg = (
                f"UnitConverter: feature '{feature}' has no registered "
                f"dimension; leaving it unscaled. Add it to "
                f"FEATURE_DIMENSIONS to convert it."
            )
            if self.strict:
                raise KeyError(msg)
            self.log_func(f"[WARNING] {msg}")
            return 1.0

        dimension = FEATURE_DIMENSIONS[feature]
        us_to_si = US_TO_SI_FACTORS[dimension]

        if self.source_system == "US" and self.target_system == "SI":
            return us_to_si
        if self.source_system == "SI" and self.target_system == "US":
            return 1.0 / us_to_si
        return 1.0

    # ------------------------------------------------------------------ #
    # Array conversion                                                    #
    # ------------------------------------------------------------------ #
    def convert_feature_vector(
        self, feature_list: List[str], feature_vector: ndarray
    ) -> ndarray:
        """Return a copy of ``feature_vector`` rescaled to ``target_system``.

        The feature axis is the *last* axis, matching
        :meth:`DatasetNormalizer.normalize_feature_vector`. Static arrays are
        shaped ``(N, F)`` and dynamic arrays ``(T, N, F)``; both are handled.

        Caveat: ``position_*`` and ``relative_position_*`` are planar
        coordinates. Scaling them by the length factor is only physically
        meaningful when the projected CRS uses the source system's linear unit
        (e.g. US survey feet for a US State Plane projection). If coordinates
        are already in metres regardless of the model's unit system, exclude
        the position features or set the source system accordingly.
        """
        if self.is_noop:
            return feature_vector

        if feature_vector.shape[-1] != len(feature_list):
            raise ValueError(
                f"Last axis of feature_vector ({feature_vector.shape[-1]}) "
                f"does not match len(feature_list) ({len(feature_list)})."
            )

        factors = np.array(
            [self.get_factor(f) for f in feature_list],
            dtype=np.float64,
        )
        # Broadcast factors across the feature (last) axis.
        converted = feature_vector.astype(np.float64) * factors
        return converted.astype(feature_vector.dtype, copy=False)

    def describe(self, feature_list: Optional[List[str]] = None) -> Dict[str, float]:
        """Return a ``{feature: factor}`` map for inspection/logging."""
        features = (
            feature_list if feature_list is not None else list(FEATURE_DIMENSIONS)
        )
        return {f: self.get_factor(f) for f in features}
