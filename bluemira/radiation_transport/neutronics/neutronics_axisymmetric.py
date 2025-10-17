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

from bluemira.base.look_and_feel import bluemira_print
from bluemira.base.parameter_frame import Parameter, ParameterFrame, make_parameter_frame
from bluemira.codes.openmc.make_csg import (
    BluemiraNeutronicsCSG,
    CellStage,
)
from bluemira.codes.openmc.material import MaterialsLibrary
from bluemira.codes.openmc.tallying import TALLY_FUNCTION_TYPE, filter_cells
from bluemira.radiation_transport.neutronics.make_pre_cell import PreCellStage
from bluemira.radiation_transport.neutronics.materials import NeutronicsMaterials
from bluemira.radiation_transport.neutronics.slicing import (
    DivertorWireAndExteriorCurve,
    PanelsAndExteriorCurve,
)

if TYPE_CHECKING:
    from pathlib import Path

    import openmc
    from matplotlib.axes import Axes
    from numpy import typing as npt

    from bluemira.base.reactor import ComponentManager
    from bluemira.geometry.wire import BluemiraWire
    from bluemira.radiation_transport.neutronics.geometry import TokamakDimensions


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
        For single null configuration, this is the only divertor
    panel_break_points:
        The start and end points for each first-wall panel
        (for N panels, the shape is (N+1, 2)).
    vacuum_vessel_inner_wire:
        interface between the inside of the vacuum vessel and the outside of the blanket
    vacuum_vessel_wire:
        The outer-boundary of the vacuum vessel
    """

    lower_divertor_inner_wire: BluemiraWire
    # For single null configuration, this is the only divertor
    panel_break_points: npt.NDArray
    vacuum_vessel_inner_wire: BluemiraWire
    vacuum_vessel_outer_wire: BluemiraWire


@dataclass
class CuttingStage:
    """Stage of making cuts to the exterior curve/ outer boundary."""

    blanket: PanelsAndExteriorCurve
    divertor: DivertorWireAndExteriorCurve

    def create_pre_cell_stage(
        self,
        snap_to_horizontal_angle: float = 45,
        blanket_discretisation: int = 10,
        divertor_discretisation: int = 5,
    ) -> PreCellStage:
        """
        Create PreCellStage

        Returns
        -------
        PreCellStage
        """
        divertor = self.divertor.make_divertor_pre_cell_array(
            discretisation_level=divertor_discretisation
        )
        first, last = divertor.exterior_vertices()[(0, -1),]

        blanket = self.blanket.make_quadrilateral_pre_cell_array(
            discretisation_level=blanket_discretisation,
            starting_cut=first[::2],
            ending_cut=last[::2],
            snap_to_horizontal_angle=snap_to_horizontal_angle,
        )

        return PreCellStage(
            blanket=blanket.straighten_exterior(preserve_volume=True), divertor=divertor
        )


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
        materials_library: NeutronicsMaterials | str | Path,
        divertor: ComponentManager | None = None,
        first_wall: ComponentManager | None = None,
        panel_points: npt.NDArray | None = None,
        material_mapping: dict | None = None,
        tally_function: TALLY_FUNCTION_TYPE | None = None,
        *,
        snap_to_horizontal_angle: float = 45,
        blanket_discretisation: int = 10,
        divertor_discretisation: int = 5,
    ):
        """
        Initialises the Neutronics Reactor.

        If geometry_type = GeometryType.SN_INTEGRATED, the arguments
        first_wall, and panel_points will not be used as they are
        considered within the blanket component.

        For custom reactors, the user has the option to provide with
        first_wall as a seperate component, with divertor considered within it
        or not.

        However, they have to make their own custom methods for cell-making or
        getting wires, the methods are defined as an abstract method here.
        """
        bluemira_print("Creating axis-symmetric neutronics model")

        self.params = make_parameter_frame(params, self.param_cls)

        if isinstance(materials_library, NeutronicsMaterials):
            self.material_library = MaterialsLibrary.from_neutronics_materials(
                materials_library
            )
        else:
            self.material_library = MaterialsLibrary.import_from_xml(
                material_mapping=material_mapping, path=materials_library
            )

        self.geometry_type = geometry_type
        self.blanket_discretisation = blanket_discretisation
        self.divertor_discretisation = divertor_discretisation
        self.snap_to_horizontal_angle = snap_to_horizontal_angle

        (self.tokamak_dimensions, self.geom) = self._get_wires_from_components(
            divertor, blanket, vacuum_vessel, first_wall, panel_points
        )

        self.tally_func = filter_cells if tally_function is None else tally_function

        if self.geometry_type == GeometryType.CUSTOM:
            self.cell_stage = self._create_cell_stage_custom()
        else:
            self.cell_stage = self._create_cell_stage_default()

        self.list_of_tallies = self._set_list_of_tallies()

    def _create_cell_stage_default(self) -> CellStage:
        """
        Cut and create Precells, converts into cell stage and set Tallies

        Returns
        -------
        CellStage
        """
        # Cutting and Pre-cell Stage making
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

        self.pre_cell_stage = cutting.create_pre_cell_stage(
            blanket_discretisation=self.blanket_discretisation,
            divertor_discretisation=self.divertor_discretisation,
            snap_to_horizontal_angle=self.snap_to_horizontal_angle,
        )

        return CellStage.from_pre_cell_stage(
            pre_cell_stage=self.pre_cell_stage,
            tokamak_dimensions=self.tokamak_dimensions,
            material_library=self.material_library,
            csg=BluemiraNeutronicsCSG(),
            control_id=True,
        )

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
    def _create_cell_stage_custom(self) -> CellStage | Any:
        """
        Customised cell making stage
        """
        ...

    @abstractmethod
    def _set_list_of_tallies(self) -> list[openmc.Tally] | None:
        """Set list of tallies"""

    @abstractmethod
    def plot_2d(self, *args, **kwargs) -> Axes:
        """
        Plot neutronics reactor 2d profile

        Returns
        -------
        :
            Axes on which the reactor is plotted.
        """
        ...
