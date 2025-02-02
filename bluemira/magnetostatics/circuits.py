# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Three-dimensional current source terms.
"""

from copy import deepcopy

import numpy as np
import numpy.typing as npt

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.geometry.coordinates import (
    Coordinates,
    in_polygon,
    rotation_matrix,
    rotation_matrix_v1v2,
)
from bluemira.magnetostatics.baseclass import CurrentSource, SourceGroup
from bluemira.magnetostatics.error import MagnetostaticsError
from bluemira.magnetostatics.polyhedral_prism import PolyhedralPrismCurrentSource
from bluemira.magnetostatics.tools import process_to_coordinates, process_xyz_array
from bluemira.magnetostatics.trapezoidal_prism import TrapezoidalPrismCurrentSource

__all__ = ["ArbitraryPlanarRectangularXSCircuit", "HelmholtzCage"]


class PlanarCircuit(SourceGroup):
    """
    Base class for a planar current loop
    """

    shape: npt.NDArray[np.float64] | Coordinates
    current: float

    def _generate_sources(
        self,
        shape: npt.NDArray[np.float64] | Coordinates,
        current: float,
        source_class: type[TrapezoidalPrismCurrentSource]
        | type[PolyhedralPrismCurrentSource],
        xs_args: npt.NDArray[np.float64],
    ) -> list[TrapezoidalPrismCurrentSource | PolyhedralPrismCurrentSource]:
        """
        Generate the sources of a given class along the discretised shape

        Returns
        -------
        sources:
            Generated sources
        """
        shape = self._process_planar_shape(shape)
        betas, alphas = self._get_betas_alphas(shape)

        normal = shape.normal_vector

        # Set up geometry, calculating all trapezoidal prism sources
        self.shape = shape.T
        self.d_l = np.diff(self.shape, axis=0)
        self.midpoints = self.shape[:-1, :] + 0.5 * self.d_l
        sources = []
        warning_called = False
        for midpoint, d_l, beta, alpha in zip(
            self.midpoints, self.d_l, betas, alphas, strict=False
        ):
            d_l_norm = d_l / np.linalg.norm(d_l)
            t_vec = np.cross(d_l_norm, normal)
            if source_class is PolyhedralPrismCurrentSource:
                source = source_class(
                    midpoint,
                    d_l,
                    normal,
                    t_vec,
                    *xs_args,
                    alpha=alpha,
                    beta=beta,
                    current=current,
                    bypass_endcap_error=True,
                    endcap_warning=False,
                )

                if warning_called is False and source._warning is True:
                    bluemira_warn(
                        "Unequal end cap angles will result in result not"
                        " being precise. This inaccuracy will increase as the"
                        " end cap angle discrepency increases."
                    )
                    warning_called = True
            else:
                source = source_class(
                    midpoint,
                    d_l,
                    normal,
                    t_vec,
                    *xs_args,
                    alpha=alpha,
                    beta=beta,
                    current=current,
                )
            sources.append(source)
        return sources

    def _process_planar_shape(
        self, shape: npt.NDArray[np.float64] | Coordinates
    ) -> Coordinates:
        """
        Checks that the shape is planar

        Returns
        -------
        shape:
            Coordinates of input shape as Coordinates object.

        Raises
        ------
        MagnetostaticsError
            Shape must be planar
        """
        shape = process_to_coordinates(shape)
        if not shape.is_planar:
            raise MagnetostaticsError(
                f"The input shape for {self.__class__.__name__} must be planar."
            )
        return shape

    def _get_betas_alphas(
        self, shape: npt.NDArray[np.float64] | Coordinates
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Get the first and second half-angles (transformed to the x-z plane)

        Returns
        -------
        :
            First and second half-angles
        """
        shape = self._transform_to_xz(deepcopy(shape))
        self._t_shape = shape
        closed = shape.closed
        self._clockwise = shape.check_ccw((0, 1, 0))
        d_l = np.diff(shape.T, axis=0)
        midpoints = shape.T[:-1, :] + 0.5 * d_l
        betas = (
            [self._get_half_angle(midpoints[-1], shape.points[0], midpoints[0])]
            if closed
            else [0.0]
        )
        alphas = []

        for i, midpoint in enumerate(midpoints):
            if i != len(midpoints) - 1:
                alpha = self._get_half_angle(
                    midpoint, shape.points[i + 1], midpoints[i + 1]
                )
            elif closed:
                alpha = self._get_half_angle(midpoint, shape.points[-1], midpoints[0])
            else:
                alpha = 0.0
            alphas.append(alpha)
            beta = alpha
            betas.append(beta)

        return np.rad2deg(betas), np.rad2deg(alphas)

    @staticmethod
    def _transform_to_xz(shape: npt.NDArray[np.float64] | Coordinates) -> Coordinates:
        """
        Transform shape to the x-z plane.

        Returns
        -------
        :
            Tranformed coordinates.
        """
        normal_vector = shape.normal_vector
        if abs(normal_vector[1]) == 1.0:
            return shape
        shape.translate(-np.array(shape.center_of_mass))

        rot_mat = rotation_matrix_v1v2(normal_vector, np.array([0.0, -1.0, 0.0]))
        return Coordinates(rot_mat @ shape._array)

    def _get_half_angle(
        self,
        p0: npt.NDArray[np.float64],
        p1: npt.NDArray[np.float64],
        p2: npt.NDArray[np.float64],
    ) -> float:
        """
        Get the half angle between three points, respecting winding direction.

        Returns
        -------
        angle:
            Half angle between three points.
        """
        v1 = p1 - p0
        v2 = p2 - p1
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        cos_angle = np.dot(v1, v2)
        angle = 0.5 * np.arccos(np.clip(cos_angle, -1, 1))

        v_norm = np.linalg.norm(v1 - v2)
        if np.isclose(v_norm, 0):
            return 0.0

        v3 = p2 - p0
        v3 /= np.linalg.norm(v3)
        project_point = p0 + np.dot(p1 - p0, v3) * v3
        d = np.linalg.norm(p1 - project_point)
        if np.isclose(d, 0.0):
            return 0.0

        point_in_poly = self._point_inside_xz(project_point)

        if not point_in_poly:
            angle *= -1

        if self._clockwise:
            angle *= -1

        if abs(angle) > 0.25 * np.pi:
            # We're actually concerned with (pi/2 - angle) < pi/4
            # If this is the case, two consecutive sources will have sharp corners that
            # will overlap
            bluemira_warn(
                f"{self.__class__.__name__} cannot handle acute angles, as there will be"
                " overlaps in the sources."
            )
        return angle

    def _point_inside_xz(self, point: npt.NDArray[np.float64]) -> bool:
        """
        Determine whether or not the point is in the polygon - accounts
        for clockwise ordered coordinates.

        Returns
        -------
        :
            Whether or not the point is in the polygon
        """
        # reverse second axis if clockwise
        ind = (slice(None), slice(None, None, -1)) if self._clockwise else slice(None)
        return in_polygon(point[0], point[2], self._t_shape.xz[ind].T)


class ArbitraryPlanarRectangularXSCircuit(PlanarCircuit):
    """
    An arbitrary, planar current loop of constant rectangular cross-section
    and uniform current density.

    Parameters
    ----------
    shape:
        The geometry from which to form an ArbitraryPlanarRectangularXSCircuit
    breadth:
        The breadth of the current source (half-width) [m]
    depth:
        The depth of the current source (half-height) [m]
    current:
        The current flowing through the source [A]

    Notes
    -----
    Works best with planar x-z geometries.
    """

    breadth: float
    depth: float

    def __init__(
        self,
        shape: npt.NDArray[np.float64] | Coordinates,
        breadth: float,
        depth: float,
        current: float,
    ):
        super().__init__(
            self._generate_sources(
                shape, current, TrapezoidalPrismCurrentSource, (breadth, depth)
            )
        )


class ArbitraryPlanarPolyhedralXSCircuit(PlanarCircuit):
    """
    An arbitrary, planar current loop of constant polyhedral cross-section
    and uniform current density.

    Parameters
    ----------
    shape:
        The geometry from which to form an ArbitraryPlanarRectangularXSCircuit
    xs_coordinates:
        The cross-section geometry of the conductor
    current:
        The current flowing through the source [A]

    Notes
    -----
    Works best with planar x-z geometries.
    """

    def __init__(
        self,
        shape: npt.NDArray[np.float64] | Coordinates,
        xs_coordinates: npt.NDArray[np.float64] | Coordinates,
        current: float,
    ):
        super().__init__(
            self._generate_sources(
                shape, current, PolyhedralPrismCurrentSource, (xs_coordinates,)
            )
        )


class HelmholtzCage(SourceGroup):
    """
    Axisymmetric arrangement of current sources about the z-axis.

    Parameters
    ----------
    circuit:
        Current source to pattern
    n_TF:
        Number of sources to pattern

    Notes
    -----
    The plane at 0 degrees is set to be between two circuits.
    """

    def __init__(self, circuit: CurrentSource, n_TF: int):
        self.n_TF = n_TF
        sources = self._pattern(circuit)

        super().__init__(sources)

    def _pattern(self, circuit: CurrentSource) -> list[CurrentSource]:
        """
        Pattern the CurrentSource axisymmetrically.

        Returns
        -------
        :
            List of arranged sources
        """
        sources = []
        for angle in np.linspace(0, 360, int(self.n_TF), endpoint=False):
            source = circuit.copy()
            source.rotate(angle, axis="z")
            sources.append(source)
        return sources

    @process_xyz_array
    def ripple(
        self,
        x: float | npt.NDArray[np.float64],
        y: float | npt.NDArray[np.float64],
        z: float | npt.NDArray[np.float64],
    ) -> float:
        """
        Get the toroidal field ripple at a point.

        Parameters
        ----------
        x:
            The x coordinate(s) of the points at which to calculate the ripple
        y:
            The y coordinate(s) of the points at which to calculate the ripple
        z:
            The z coordinate(s) of the points at which to calculate the ripple

        Returns
        -------
        :
            The value of the TF ripple at the point(s) [%]
        """
        point = np.array([x, y, z])
        ripple_field = np.zeros(2)
        n = np.array([0, 1, 0])
        planes = [0, np.pi / self.n_TF]  # rotate (inline, ingap)

        for i, theta in enumerate(planes):
            r_matrix = rotation_matrix(theta).T
            sr = np.dot(point, r_matrix)
            nr = np.dot(n, r_matrix)
            field = self.field(*sr)
            ripple_field[i] = np.dot(nr, field)

        return 1e2 * (ripple_field[0] - ripple_field[1]) / np.sum(ripple_field)
