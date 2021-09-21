# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
Optimisation variable class.
"""

import numpy as np
import matplotlib.pyplot as plt

from bluemira.utilities.error import OptVariablesError
from bluemira.base.constants import BLUEMIRA_PAL_MAP


def normalise_value(value, lower_bound, upper_bound):
    """
    Normalise a value uniformly [0 -> 1].

    Parameters
    ----------
    value: float
        Value to normalise
    lower_bound: float
        Lower bound at which to normalise
    upper_bound: float
        Upper bound at which to normalise

    Returns
    -------
    v_norm: float
        Normalised value [0 -> 1]
    """
    return (value - lower_bound) / (upper_bound - lower_bound)


def denormalise_value(v_norm, lower_bound, upper_bound):
    """
    Denormalise a value uniformly from [0 -> 1] w.r.t bounds.

    Parameters
    ----------
    v_norm: float
        Normalised value
    lower_bound: float
        Lower bound at which to denormalise
    upper_bound: float
        Upper bound at which to denormalise

    Returns
    -------
    value: float
        Denormalised value w.r.t. bounds
    """
    return lower_bound + v_norm * (upper_bound - lower_bound)


class BoundedVariable:
    """
    A bounded variable, uniformly normalised from 0 to 1 w.r.t. its bounds.

    Parameters
    ----------
    name: str
        Name of the variable
    value: float
        Value of the variable
    lower_bound: float
        Lower bound of the variable
    upper_bound: float
        Upper bound of the variable
    fixed: bool
        Whether or not the variable is to be held constant
    """

    __slots__ = ("name", "lower_bound", "upper_bound", "fixed", "_value")

    def __init__(self, name, value, lower_bound, upper_bound, fixed=False):
        self._validate_bounds(lower_bound, upper_bound)
        self._validate_value(value, lower_bound, upper_bound)
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self._value = None
        self.fixed = False  # Required to set value initially
        self.value = value
        self.fixed = fixed

    @property
    def value(self):
        """
        The value of the variable.
        """
        return self._value

    @value.setter
    def value(self, value):
        """
        Set the value of the variable, enforcing bounds.
        """
        if self.fixed:
            raise OptVariablesError("Cannot set the value of a fixed variable.")

        self._validate_value(value, self.lower_bound, self.upper_bound)
        self._value = value

    def fix(self, value: float):
        """
        Fix the variable at a specified value. Ignores bounds.

        Parameters
        ----------
        value: float
            Value at which to fix the variable.
        """
        self.fixed = True
        if value is not None:
            self._value = value

    def adjust(self, value=None, lower_bound=None, upper_bound=None):
        """
        Adjust the BoundedVariable.

        Parameters
        ----------
        name: str
            Name of the variable to adjust
        value: Optional[float]
            Value of the variable to set
        lower_bound: Optional[float]
            Value of the lower bound to set
        upper_bound: Optional[float]
            Value of the upper to set
        """
        if self.fixed:
            raise OptVariablesError("Cannot adjust a fixed variable.")
        if lower_bound is not None:
            self.lower_bound = lower_bound
        if upper_bound is not None:
            self.upper_bound = upper_bound

        self._validate_bounds(self.lower_bound, self.upper_bound)

        if value is not None:
            self.value = value

    @property
    def normalised_value(self) -> float:
        """
        The normalised value of the variable.
        """
        return normalise_value(self.value, self.lower_bound, self.upper_bound)

    @staticmethod
    def _validate_bounds(lower_bound, upper_bound):
        if lower_bound > upper_bound:
            raise OptVariablesError("Lower bound is higher than upper bound.")

    @staticmethod
    def _validate_value(value, lower_bound, upper_bound):
        if not lower_bound <= value <= upper_bound:
            raise OptVariablesError("Variable value is out of its bounds.")


class OptVariables:
    """
    A set of ordered variables to facilitate optimisation using normalised values.

    Parameters
    ----------
    variables: List[BoundedVariable]
        Set of variables to use
    """

    def __init__(self, variables):
        self._var_dict = {v.name: v for v in variables}

    def add_variable(self, variable):
        """
        Add a variable to the set.

        Parameters
        ----------
        variable: BoundedVariable
            Variable to add to the set.
        """
        if variable.name in self._var_dict:
            raise OptVariablesError(f"Variable {variable.name} already in OptVariables.")

        self._var_dict[variable.name] = variable

    def remove_variable(self, name):
        """
        Remove a variable from the set.

        Parameters
        ----------
        name: str
            Name of the variable to remove
        """
        self._check_presence(name)

        del self._var_dict[name]

    def adjust_variable(self, name, value=None, lower_bound=None, upper_bound=None):
        """
        Adjust a variable in the set.

        Parameters
        ----------
        name: str
            Name of the variable to adjust
        value: Optional[float]
            Value of the variable to set
        lower_bound: Optional[float]
            Value of the lower bound to set
        upper_bound: Optional[float]
            Value of the upper to set
        """
        self._check_presence(name)
        self._var_dict[name].adjust(value, lower_bound, upper_bound)

    def fix_variable(self, name, value=None):
        """
        Fix a variable in the set, removing it from optimisation but preserving a
        constant value.

        Parameters
        ----------
        name: str
            Name of the variable to fix
        value: Optional[float]
            Value at which to fix the variable (will default to present value)
        """
        self._check_presence(name)

        self._var_dict[name].fix(value=value)

    def get_normalised_values(self):
        """
        Get the normalised values of all free variables.

        Returns
        -------
        x_norm: np.ndarray
            Array of normalised values
        """
        return np.array(
            [v.normalised_value for v in self._var_dict.values() if not v.fixed]
        )

    def set_values_from_norm(self, x_norm):
        """
        Set values from a normalised vector.

        Parameters
        ----------
        x_norm: np.ndarray
            Array of normalised values
        """
        if len(x_norm) != self.n_free_variables:
            raise OptVariablesError(
                f"Number of normalised variables {len(x_norm)} != {self.n_free_variables}."
            )

        for name, v_norm in zip(self._opt_vars, x_norm):
            variable = self._var_dict[name]
            value = denormalise_value(v_norm, variable.lower_bound, variable.upper_bound)
            variable.value = value

    @property
    def values(self):
        """
        All un-normalised values of the variable set (including fixed variable values).
        """
        return np.array([v.value for v in self._var_dict.values()])

    @property
    def n_free_variables(self) -> int:
        """
        Number of free variables in the set.
        """
        return len(self._opt_vars)

    @property
    def _opt_vars(self):
        return [v.name for v in self._var_dict.values() if not v.fixed]

    def _check_presence(self, name):
        if name not in self._var_dict.keys():
            raise OptVariablesError(f"Variable {name} not in OptVariables.")

    def plot(self):
        """
        Plot the OptVariables.
        """
        _, ax = plt.subplots()
        left_labels = [
            f"{v.name}: {v.lower_bound:.2f} " for v in self._var_dict.values()
        ]
        right_labels = [f"{v.upper_bound:.2f}" for v in self._var_dict.values()]
        y_pos = np.arange(len(left_labels))

        x_norm = [
            v.normalised_value if not v.fixed else 0.5 for v in self._var_dict.values()
        ]
        colors = [
            BLUEMIRA_PAL_MAP["red"] if v.fixed else BLUEMIRA_PAL_MAP["blue"]
            for v in self._var_dict.values()
        ]

        values = [f"{v:.2f}" for v in self.values]
        ax2 = ax.twinx()
        ax.barh(y_pos, x_norm, color="w")
        ax2.barh(y_pos, x_norm, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(left_labels)
        ax.invert_yaxis()

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(right_labels)
        ax2.invert_yaxis()
        ax.set_xlim([-0.1, 1.1])
        ax.set_xlabel("$x_{norm}$")

        for xi, yi, vi in zip(x_norm, y_pos, values):
            ax.text(xi, yi, vi)