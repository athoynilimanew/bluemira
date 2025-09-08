# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Axis-symmetric CSG CAD models for neutronics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import numpy as np

from bluemira.base.look_and_feel import bluemira_print
from bluemira.base.parameter_frame import Parameter, ParameterFrame, make_parameter_frame
from bluemira.codes.openmc.make_csg import BluemiraNeutronicsCSG, make_cell_arrays
from bluemira.codes.openmc.material import MaterialsLibrary
from bluemira.geometry.plane import calculate_plane_dir
from bluemira.geometry.tools import get_wire_plane_intersect, make_polygon
from bluemira.radiation_transport.neutronics.make_pre_cell import PreCell
from bluemira.radiation_transport.neutronics.slicing import (
    DivertorWireAndExteriorCurve,
    PanelsAndExteriorCurve,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy import typing as npt

    from bluemira.base.reactor import ComponentManager
    from bluemira.codes.openmc.make_csg import CellStage
    from bluemira.geometry.wire import BluemiraWire
    from bluemira.radiation_transport.neutronics.geometry import TokamakDimensions
    from bluemira.radiation_transport.neutronics.make_pre_cell import (
        DivertorPreCellArray,
        PreCellArray,
    )
    from bluemira.radiation_transport.neutronics.materials import NeutronicsMaterials


class GeometryType(Enum):
    """Enumeration of geometry types."""

    # Single-null geometry with integrated FW & blanket, plus divertor and vessel
    SN_INTEGRATED = auto()

    # Other
    CUSTOM = auto()

    @classmethod
    def _missing_(cls, value: str):
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(
                f"{cls.__name__} has no type {value}"
                f"please select from {(*cls._member_names_,)}"
            ) from None


@dataclass
class ReactorGeometry:
    """
    Data storage stage

    Parameters
    ----------
    lower_divertor_inner_wire:
        The plasma-facing side of the divertor.
    panel_break_points:
        The start and end points for each first-wall panel
        (for N panels, the shape is (N+1, 2)).
    vacuum_vessel_inner_wire:
        interface between the inside of the vacuum vessel and the outside of the blanket
    vacuum_vessel_wire:
        The outer-boundary of the vacuum vessel
    customised_geometry:
        only useful in case of user-defined customised reactor
    """

    lower_divertor_inner_wire: BluemiraWire
    panel_break_points: npt.NDArray
    vacuum_vessel_inner_wire: BluemiraWire
    vacuum_vessel_outer_wire: BluemiraWire
    customised_geometry: Any | None = None


@dataclass
class CuttingStage:
    """Stage of making cuts to the exterior curve/ outer boundary."""

    blanket: PanelsAndExteriorCurve
    divertor: DivertorWireAndExteriorCurve


class PreCellStage:
    """Stage of making pre-cells"""

    def __init__(self, blanket: PreCellArray, divertor: DivertorPreCellArray):
        """Check convexity after initialization"""
        self.blanket = blanket.copy()
        self.divertor = divertor
        # 1. stretch first blanket cell in PreCellArray to reach div_start_wire
        div_start_wire = self.divertor[0].cw_wall.restore_to_wire()
        # pull everything down to: div_start_wire.
        # Alternatively, choose div_start_wire=self.divertor[0].outline
        old_vv_wire = self.blanket[0].vv_wire
        ext_pt, i_high, i_low = np.insert(self.blanket[0].vertex, 1, 0, axis=0).T[:3]
        i_end = get_wire_plane_intersect(
            div_start_wire, *calculate_plane_dir(i_high, i_low)
        )
        # v_end = get_wire_plane_intersect(
        #     div_start_wire,
        #     *calculate_plane_dir(old_vv_wire.end_point().xyz.flatten(),
        #     old_vv_wire.start_point().xyz.flatten())
        # )
        in_wire = make_polygon(np.array([i_high, i_end]).T, closed=False)
        vv_wire = make_polygon(
            np.array([
                self.divertor[0].vv_wire.end_point,
                old_vv_wire.end_point().xyz.flatten(),
            ]).T,
            closed=False,
        )
        ex_wire = make_polygon(
            np.array([self.divertor[0].vertex.T[0], ext_pt]).T, closed=False
        )
        new_start_cell = PreCell(in_wire, vv_wire, ex_wire)
        self.blanket[0] = new_start_cell

        # 2. stretch first blanket cell in PreCellArray to reach div_end_wire
        div_end_wire = self.divertor[-1].ccw_wall.restore_to_wire()
        old_vv_wire = self.blanket[-1].vv_wire
        i_low, i_high, ext_pt = np.insert(self.blanket[-1].vertex, 1, 0, axis=0).T[-3:]
        i_start = get_wire_plane_intersect(
            div_end_wire, *calculate_plane_dir(i_high, i_low)
        )
        # v_end = get_wire_plane_intersect(
        #     div_end_wire,
        #     *calculate_plane_dir(old_vv_wire.start_point().xyz.flatten(),
        #     old_vv_wire.end_point().xyz.flatten())
        # )
        in_wire = make_polygon(np.array([i_start, i_high]).T, closed=False)
        vv_wire = make_polygon(
            np.array([
                old_vv_wire.start_point().xyz.flatten(),
                self.divertor[-1].vv_wire.start_point,
            ]).T,
            closed=False,
        )
        ex_wire = make_polygon(
            np.array([ext_pt, self.divertor[-1].vertex.T[-1]]).T, closed=False
        )
        new_end_cell = PreCell(in_wire, vv_wire, ex_wire)
        self.blanket[-1] = new_end_cell

        # re-initialize so that the cell_walls are re-calculated
        self.blanket = self.blanket.copy()

    def external_coordinates(self) -> npt.NDArray:
        """
        Get the outermost coordinates of the tokamak cross-section from pre-cell array
        and divertor pre-cell array.
        Runs clockwise, beginning at the inboard blanket-divertor joint.

        Returns
        -------
        :
            All vertices on the exterior of the blanket and the divertor.
        """
        return np.concatenate([
            self.blanket.exterior_vertices(),
            self.divertor.exterior_vertices()[::-1],
        ])

    def bounding_box(self) -> tuple[float, ...]:
        """
        Get bounding box of pre cell stage

        Returns
        -------
        z_max:
            The maximum height of the bounding box.
        z_min:
            The minimum height of the bounding box.
        r_max:
            The maximum major radius reached by the pre-cells.
        -r_max:
            The minimum major radius reached by the pre-cells. Due to axial symmetry,
            this must be =-r_max.
        """
        all_ext_vertices = self.external_coordinates()
        z_min = all_ext_vertices[:, -1].min()
        z_max = all_ext_vertices[:, -1].max()
        r_max = max(abs(all_ext_vertices[:, 0]))
        return z_max, z_min, r_max, -r_max

    def half_bounding_box(self) -> tuple[float, ...]:
        """
        Get bounding box of the 2D poloidal cross-section of the right-hand half of the
        reactor.

        Returns
        -------
        z_max:
            The maximum height of the bounding box.
        z_min:
            The minimum height of the bounding box.
        r_max:
            The maximum major radius reached by the pre-cells.
        r_min:
            The minimum major radius reached by the pre-cells on one side of the xz cross
            section. This is typically non-zero because pre-cells should not cross the
            z-axis of symmetry.
        """
        all_ext_vertices = self.external_coordinates()
        z_min = all_ext_vertices[:, -1].min()
        z_max = all_ext_vertices[:, -1].max()
        r_max = max(abs(all_ext_vertices[:, 0]))
        r_min = min(abs(all_ext_vertices[:, 0]))
        return z_max, z_min, r_max, r_min


@dataclass
class NeutronicsReactorParameterFrame(ParameterFrame):
    """Neutronics reactor parameters"""

    inboard_fw_tk: Parameter[float]
    inboard_breeding_tk: Parameter[float]
    outboard_fw_tk: Parameter[float]
    outboard_breeding_tk: Parameter[float]
    r_tf_in: Parameter[float]
    tk_tf_inboard: Parameter[float]
    fw_divertor_surface_tk: Parameter[float]
    fw_blanket_surface_tk: Parameter[float]
    blk_ib_manifold: Parameter[float]
    tk_rs: Parameter[float]
    blk_ob_manifold: Parameter[float]


class NeutronicsReactor(ABC):
    """Pre csg cell reactor"""

    param_cls = NeutronicsReactorParameterFrame

    def __init__(
        self,
        geometry_type: GeometryType,
        params: dict | ParameterFrame,
        blanket: ComponentManager,
        vacuum_vessel: ComponentManager,
        materials_library: NeutronicsMaterials,
        divertor: ComponentManager | None = None,
        first_wall: ComponentManager | None = None,
        panel_points: npt.NDArray | None = None,
        *,
        snap_to_horizontal_angle: float = 45,
        blanket_discretisation: int = 10,
        divertor_discretisation: int = 5,
    ):
        bluemira_print("Creating axis-symmetric neutronics model")

        self.params = make_parameter_frame(params, self.param_cls)
        self.material_library = MaterialsLibrary.from_neutronics_materials(
            materials_library
        )
        self.geometry_type = geometry_type
        (self.tokamak_dimensions, self.geom) = self._get_wires_from_components(
            divertor, blanket, vacuum_vessel, first_wall, panel_points
        )

        self.cell_arrays = self._create_cell_stage(
            blanket_discretisation=blanket_discretisation,
            divertor_discretisation=divertor_discretisation,
            snap_to_horizontal_angle=snap_to_horizontal_angle,
        )

    def _create_cell_stage(
        self, blanket_discretisation, divertor_discretisation, snap_to_horizontal_angle
    ) -> CellStage:
        if self.geometry_type == GeometryType.CUSTOM:
            # User's own method
            return self._create_cell_stage_custom(
                blanket_discretisation=blanket_discretisation,
                divertor_discretisation=divertor_discretisation,
                snap_to_horizontal_angle=snap_to_horizontal_angle,
                control_id=True,
            )

        cutting = CuttingStage(
            blanket=PanelsAndExteriorCurve(
                self.geom.panel_break_points,
                self.geom.vacuum_vessel_inner_wire,
                self.geom.vacuum_vessel_outer_wire,
            ),
            divertor=DivertorWireAndExteriorCurve(
                self.geom.lower_divertor_inner_wire,
                self.geom.vacuum_vessel_inner_wire,
                self.geom.vacuum_vessel_outer_wire,
            ),
        )
        divertor = cutting.divertor.make_divertor_pre_cell_array(
            discretisation_level=divertor_discretisation
        )
        first, last = divertor.exterior_vertices()[(0, -1),]

        blanket = cutting.blanket.make_quadrilateral_pre_cell_array(
            discretisation_level=blanket_discretisation,
            starting_cut=first[::2],
            ending_cut=last[::2],
            snap_to_horizontal_angle=snap_to_horizontal_angle,
        )

        pre_cell_stage = PreCellStage(
            blanket=blanket.straighten_exterior(preserve_volume=True), divertor=divertor
        )

        # setup importnat properties from precell stage
        self.bounding_box = pre_cell_stage.bounding_box()
        self.half_bounding_box = pre_cell_stage.half_bounding_box()
        self.blanket_pre_cell_array = pre_cell_stage.blanket
        self.divertor_pre_cell_array = pre_cell_stage.divertor

        return make_cell_arrays(
            pre_cell_stage=pre_cell_stage,
            csg=BluemiraNeutronicsCSG(),
            materials=self.material_library,
            tokamak_dimensions=self.tokamak_dimensions,
            control_id=True,
        )

    def plot_2d(self, *args, **kwargs) -> Axes:
        """
        Plot neutronics reactor 2d profile

        Returns
        -------
        :
            Axes on which the reactor is plotted.
        """
        if self.geometry_type == GeometryType.CUSTOM:
            # User's own method
            return self.plot_2d_custom(*args, **kwargs)

        show = kwargs.pop("show", True)
        ax = kwargs.pop("ax", None)
        ax = self.blanket_pre_cell_array.plot_2d(*args, ax=ax, show=False, **kwargs)
        return self.divertor_pre_cell_array.plot_2d(*args, ax=ax, show=show, **kwargs)

    @abstractmethod
    def _get_wires_from_components(
        self,
        divertor: ComponentManager | None = None,
        blanket: ComponentManager | None = None,
        vacuum_vessel: ComponentManager | None = None,
        first_wall: ComponentManager | None = None,
        panel_points: npt.NDArray | None = None,
    ) -> tuple[TokamakDimensions, ReactorGeometry]:
        """Get wires from components"""
        ...

    @abstractmethod
    def _create_cell_stage_custom(
        self,
        blanket_discretisation: int,
        divertor_discretisation: int,
        snap_to_horizontal_angle: float,
        *,
        control_id: bool = False,
    ) -> CellStage:
        """Customised Pre-cell making stage"""
        ...

    @abstractmethod
    def plot_2d_custom(self, *args, **kwargs) -> Axes:
        """
        Plot neutronics reactor 2d profile, customised
        """
        ...
