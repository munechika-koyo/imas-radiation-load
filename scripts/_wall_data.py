"""Read measured wall subsets and symmetric standard errors into in-memory VTK data."""

from __future__ import annotations

import numpy as np
from imas import DBEntry
from imas_paraview.io.read_geom import (
    convert_grid_subset_geometry_to_unstructured_grid,
    fill_vtk_points,
)
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkPoints, vtkStringArray
from vtkmodules.vtkCommonDataModel import vtkCompositeDataSet, vtkMultiBlockDataSet


def read_wall_results(uri: str, occurrence: int = 0) -> vtkMultiBlockDataSet:
    """Load a workflow output, preserving subset element order and skipping unmeasured faces.

    Symmetric IMAS error bars may omit values_error_lower. The absolute error
    array is the positive one-sigma deviation, not the upper confidence bound.
    """
    with DBEntry(uri, "r") as entry:
        wall = entry.get("wall", occurrence=occurrence, autoconvert=False)
    if len(wall.time) != 1 or len(wall.description_ggd) != 1:
        raise ValueError("Wall previews require one output wall description and time slice")
    description = wall.description_ggd[0]
    if len(description.grid_ggd) != 1 or len(description.ggd) != 1:
        raise ValueError("Wall previews require one output wall grid and GGD time slice")
    grid = description.grid_ggd[0]
    subsets = {int(subset.identifier.index): i for i, subset in enumerate(grid.grid_subset)}
    if len(subsets) != len(grid.grid_subset):
        raise ValueError("Wall grid_subset identifiers must be unique")
    points = vtkPoints()
    points.SetDataTypeToDouble()
    fill_vtk_points(grid, 0, points, "wall")
    output = vtkMultiBlockDataSet()
    time = numpy_to_vtk(np.asarray(wall.time), deep=True)
    time.SetName("Time")
    output.GetFieldData().AddArray(time)
    measured: set[int] = set()
    names: set[str] = set()
    for quantity in description.ggd[0].power_density:
        mean = np.asarray(quantity.values)
        error = np.asarray(quantity.values_error_upper)
        lower = np.asarray(quantity.values_error_lower)
        if not mean.size:
            if error.size or lower.size:
                raise ValueError("An unmeasured wall subset must have empty error bars")
            continue
        subset_id = int(quantity.grid_subset_index)
        if int(quantity.grid_index) != 1 or subset_id not in subsets or subset_id in measured:
            raise ValueError("Invalid or duplicate power_density grid/subset mapping")
        measured.add(subset_id)
        subset_index = subsets[subset_id]
        subset = grid.grid_subset[subset_index]
        name = str(subset.identifier.name)
        if not name or name in names:
            raise ValueError("Measured wall subsets must have unique nonempty names")
        names.add(name)
        if (
            int(subset.dimension) != 3
            or mean.shape != (len(subset.element),)
            or error.shape != mean.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(error).all()
            or (mean < 0).any()
            or (error < 0).any()
        ):
            raise ValueError(f"{name}: expected finite nonnegative face values and standard errors")
        if lower.size and not np.array_equal(lower, error):
            raise ValueError(f"{name}: wall previews require symmetric standard errors")
        data = convert_grid_subset_geometry_to_unstructured_grid(grid, subset_index, points)
        if data.GetNumberOfCells() != mean.size:
            raise ValueError(f"{name}: wall geometry does not match the number of face values")
        percent = np.full(mean.shape, np.nan)
        np.divide(error, np.abs(mean), out=percent, where=mean != 0)
        percent *= 100
        for field, values in (
            ("PowerDensity", mean),
            ("PowerDensityError", error),
            ("RelativeErrorPercent", percent),
        ):
            array = numpy_to_vtk(values, deep=True)
            array.SetName(field)
            data.GetCellData().AddArray(array)
        index = output.GetNumberOfBlocks()
        component = vtkStringArray()
        component.SetName("Component")
        component.InsertNextValue(name)
        data.GetFieldData().AddArray(component)
        output.SetBlock(index, data)
        output.GetMetaData(index).Set(vtkCompositeDataSet.NAME(), name)
    if not measured:
        raise ValueError("Wall IDS has no measured power_density subsets")
    return output
