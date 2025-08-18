# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Test script to make the CSG branch work."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bluemira.codes.wrapper import neutronics_code_solver
from bluemira.radiation_transport.error import NeutronicsError
from bluemira.radiation_transport.neutronics.blanket_data import (
    create_materials,
    get_preset_physical_properties,
)
from bluemira.radiation_transport.neutronics.geometry import TokamakDimensions
from bluemira.radiation_transport.neutronics.neutronics_axisymmetric import (
    GeometryType,
    NeutronicsReactor,
    NeutronicsReactorParameterFrame,
    ReactorGeometry,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import openmc.source
    from numpy import typing as npt

    from bluemira.base.parameter_frame import ParameterFrame
    from bluemira.base.reactor import ComponentManager
    from bluemira.codes.openmc.output import OpenMCResult
    from bluemira.codes.openmc.params import PlasmaSourceParameters
    from eudemo.blanket import Blanket
    from eudemo.ivc import IVCShapes
    from eudemo.vacuum_vessel import VacuumVessel


class EUDEMONeutronicsCSGReactor(NeutronicsReactor):
    """EUDEMO Axis-symmetric neutronics model"""

    def _get_wires_from_components(
        self,
        ivc_shapes: IVCShapes,
        blanket: Blanket,
        vacuum_vessel: VacuumVessel,
        panel_points: npt.NDArray,
    ) -> tuple[TokamakDimensions, ReactorGeometry]:
        return (
            TokamakDimensions.from_parameterframe(self.params, blanket.r_inner_cut),
            ReactorGeometry(
                divertor_wires=(ivc_shapes.div_internal_boundary, None),
                panel_break_points=panel_points,
                vacuum_vessel_inner_wire=ivc_shapes.outer_boundary,
                vacuum_vessel_outer_wire=vacuum_vessel.xz_boundary,
            ),
        )


def run_neutronics(
    params: dict | ParameterFrame,
    build_config: dict,
    blanket: ComponentManager,
    vacuum_vessel: ComponentManager,
    ivc_shapes: IVCShapes,
    source: Callable[[PlasmaSourceParameters], openmc.source.SourceBase] | None = None,
    tally_function=None,
) -> tuple[EUDEMONeutronicsCSGReactor, OpenMCResult | dict[int, float]]:
    """Runs the neutronics model

    Returns
    -------
    neutronics_csg:
        The neutronics CSG reactor model
    res:
        The result of the neutronics run

    Raises
    ------
    NeutronicsError
        Can't import default neutron source
    """
    # TODO get these materials from the componentmanager or something similar
    breeder_materials, tokamak_geometry = get_preset_physical_properties(
        build_config.pop("blanket_type")
    )
    material_library = create_materials(breeder_materials)

    csg_params = NeutronicsReactorParameterFrame.from_config_params(params)
    csg_params.update_from_dict(
        {
            "inboard_fw_tk": {"value": tokamak_geometry.inb_fw_thick, "unit": "m"},
            "inboard_breeding_tk": {"value": tokamak_geometry.inb_bz_thick, "unit": "m"},
            "outboard_fw_tk": {"value": tokamak_geometry.outb_fw_thick, "unit": "m"},
            "outboard_breeding_tk": {
                "value": tokamak_geometry.outb_bz_thick,
                "unit": "m",
            },
        },
        source="Neutronics",
    )

    # TODO: Directly make cell arrays
    neutronics_csg = EUDEMONeutronicsCSGReactor(
        geometry_type=GeometryType.SN_INTEGRATED,
        params=csg_params,
        divertor=ivc_shapes,
        blanket=blanket,
        vacuum_vessel=vacuum_vessel,
        materials_library=material_library,
        panel_points=blanket.panel_points.T,
    )

    if source is None:
        try:
            from bluemira.codes.openmc.sources import make_pps_source  # noqa: PLC0415
        except ImportError:
            raise NeutronicsError("Cannot import neutronics source") from None

    solver = neutronics_code_solver(
        params,
        build_config,
        neutronics_csg.cell_arrays,
        source=source or make_pps_source,
        tally_function=tally_function,
    )

    res = solver.execute()

    return neutronics_csg, res
