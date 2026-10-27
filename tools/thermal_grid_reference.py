"""Transparent two-room algebra fixture; no scenario loading or env rollouts.

Dense assembly is independent of the simulator's assembly/elimination helpers.
The source solver is used only as the object under test. All fixture constants
are synthetic; none are inferred properties of a competition building.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import scipy.sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smart_buildings.smart_control.simulator import building, finite_difference_solver


def fixture(*, room_rows=3, room_columns=2):
    if (not isinstance(room_rows, int) or not isinstance(room_columns, int)
            or not 2 <= room_rows <= 12 or not 2 <= room_columns <= 12):
        raise ValueError('Fixture room dimensions must be integers in 2..12')
    shape = (room_rows + 4, 2 * room_columns + 5)
    last_row, last_column = shape[0] - 2, shape[1] - 2
    partition_column = room_columns + 2
    inside = np.zeros(shape, dtype=bool)
    inside[1:last_row + 1, 1:last_column + 1] = True
    rooms = [np.zeros(shape, dtype=bool) for _ in range(2)]
    rooms[0][2:last_row, 2:partition_column] = True
    rooms[1][2:last_row, partition_column + 1:last_column] = True
    air = rooms[0] | rooms[1]
    conductivity = np.full(shape, .4)
    conductivity[2:last_row, partition_column] = .8
    conductivity[air] = 20.
    density = np.full(shape, 800.)
    density[air] = 1.2
    cp = np.full(shape, 900.)
    cp[air] = 1006.
    neighbors = [[[(r + dr, c + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                   if 0 <= r + dr < shape[0] and 0 <= c + dc < shape[1]
                   and inside[r + dr, c + dc]] if inside[r, c] else []
                  for c in range(shape[1])] for r in range(shape[0])]
    field = SimpleNamespace(temp=np.full(shape, 295.), input_q=np.zeros(shape),
        cv_size_cm=20., floor_height_cm=300., conductivity=conductivity,
        density=density, heat_capacity=cp, neighbors=neighbors,
        interior_mass=building.InteriorMass(
            building.MaterialProperties(conductivity=12., heat_capacity=1000., density=30.), air))
    coords = list(zip(*np.where(inside)))
    index = {coordinate: i for i, coordinate in enumerate(coords)}
    mass_coords = list(zip(*np.where(air)))
    mass_index = {coordinate: len(coords) + i for i, coordinate in enumerate(mass_coords)}
    count = len(coords) + len(mass_coords)
    capacity, exposure = np.zeros(count), np.zeros(count)
    laplacian = np.zeros((count, count))
    widths = {p: (.1 if p[1] in (1, last_column) else .2,
                  .1 if p[0] in (1, last_row) else .2) for p in coords}

    def link(i, j, conductance):
        laplacian[i, i] += conductance
        laplacian[j, j] += conductance
        laplacian[i, j] -= conductance
        laplacian[j, i] -= conductance

    film = 8.
    for (r, c), i in index.items():
        width, height = widths[r, c]
        capacity[i] = width * height * 3 * density[r, c] * cp[r, c]
        for dr, dc in ((1, 0), (0, 1)):
            other = (r + dr, c + dc)
            if other not in index:
                continue
            ow, oh = widths[other]
            face = 3 * min(width, ow) if dr else 3 * min(height, oh)
            thickness, other_thickness = (height, oh) if dr else (width, ow)
            conductance = face / (thickness / (2 * conductivity[r, c])
                                 + other_thickness / (2 * conductivity[other]))
            link(i, index[other], conductance)
        if r in (1, last_row):
            exposure[i] += 3 * width / (height / (2 * conductivity[r, c]) + 1 / film)
        if c in (1, last_column):
            exposure[i] += 3 * height / (width / (2 * conductivity[r, c]) + 1 / film)
        if (r, c) in mass_index:
            j = mass_index[r, c]
            capacity[j] = 30 * 1000 * width * height * 3
            link(i, j, 12 * width * height / 3)

    applied, sensed, uniform = [np.zeros((count, 2)) for _ in range(3)]
    for room, mask in enumerate(rooms):
        members = [index[p] for p in zip(*np.where(mask))]
        applied[members[len(members) // 2], room] = 1.
        sensed[members, room] = 1 / len(members)
        uniform[members, room] = 1 / len(members)
    return SimpleNamespace(field=field, coords=coords, index=index, mass_index=mass_index,
        rooms=rooms, capacity=capacity, exposure=exposure, laplacian=laplacian,
        applied=applied, sensed=sensed, uniform=uniform, film=film, seconds=300.,
        partition_column=partition_column, last_row=last_row, last_column=last_column)


def solve_reference(grid, previous, *, ambient=275., fixed=(3., 2.),
                    conductance=(4., 3.), supply=(285., 289.), basis=None):
    """Full air/wall/slab backward Euler, optionally constrained to a basis.

    Piecewise-constant columns preserve aggregate energy but discard spatial
    modes. Projecting the old state is capacity-weighted, not a plain mean.
    """
    rate = grid.capacity / grid.seconds
    matrix = (np.diag(rate + grid.exposure) + grid.laplacian
              + (grid.applied * conductance) @ grid.sensed.T)
    rhs = (rate * previous + grid.exposure * ambient + grid.uniform @ fixed
           + grid.applied @ (np.asarray(conductance) * supply))
    if basis is None:
        return np.linalg.solve(matrix, rhs)
    return basis @ np.linalg.solve(basis.T @ matrix @ basis, basis.T @ rhs)


def reduced_basis(grid):
    """Lump each room's air/slab separately; retain every wall CV explicitly."""
    groups = []
    for mask in grid.rooms:
        coords = list(zip(*np.where(mask)))
        groups.append([grid.index[p] for p in coords])
        groups.append([grid.mass_index[p] for p in coords])
    air_coords = set(grid.mass_index)
    groups.extend([[i] for p, i in grid.index.items() if p not in air_coords])
    basis = np.zeros((len(grid.capacity), len(groups)))
    for column, members in enumerate(groups):
        basis[members, column] = 1.
    return basis


def solve_source(grid, previous, *, ambient=275., fixed=(3., 2.),
                 conductance=(4., 3.), supply=(285., 289.)):
    """A single source-solver unit invocation; no environment or policy."""
    field = grid.field
    temp = np.full(field.temp.shape, ambient)
    for coordinate, i in grid.index.items():
        temp[coordinate] = previous[i]
    solver = finite_difference_solver.FiniteDifferenceSolver(field, grid.seconds)
    # Explicitly initialize arbitrary latent history for this algebra test.
    # Private state is used ONLY in this synthetic reference, never a policy.
    for coordinate, j in grid.mass_index.items():
        solver._mass_temp[grid.index[coordinate]] = previous[j]
    p, s = [np.zeros((temp.size, 2)) for _ in range(2)]
    power = np.zeros_like(temp)
    for coordinate, i in grid.index.items():
        flat = np.ravel_multi_index(coordinate, temp.shape)
        p[flat], s[flat] = grid.applied[i], grid.sensed[i]
        power[coordinate] = (grid.uniform @ fixed)[i]
    solver.couple_air_side(scipy.sparse.csc_matrix(p), scipy.sparse.csc_matrix(s))
    updated = solver.solve(temp, power, ambient, grid.film,
                           np.asarray(conductance), np.asarray(supply))
    return np.r_[[updated[p] for p in grid.coords], solver.interior_mass_temp]


def energy_audit(grid, previous, updated, *, ambient=275., fixed=(3., 2.),
                 conductance=(4., 3.), supply=(285., 289.)):
    air_power = np.asarray(conductance) * (supply - grid.sensed.T @ updated)
    storage = grid.capacity * (updated - previous) / grid.seconds
    source = grid.uniform @ fixed + grid.applied @ air_power
    outside = grid.exposure * (ambient - updated)
    residual = storage - source - outside + grid.laplacian @ updated
    # Coarse energy partitions are diagnostics, independent of the reduction
    # basis: air/slab per room, shared partition, facade.
    masks, labels = [], []
    for room, mask in enumerate(grid.rooms):
        coords = list(zip(*np.where(mask)))
        for kind, mapping in [('air', grid.index), ('slab', grid.mass_index)]:
            indices = [mapping[p] for p in coords]
            masks.append(indices)
            labels.append(f'room_{room + 1}_{kind}')
    walls = [(p, i) for p, i in grid.index.items() if p not in grid.mass_index]
    def is_partition(p):
        return p[1] == grid.partition_column and 2 <= p[0] < grid.last_row
    masks += [[i for p, i in walls if is_partition(p)],
              [i for p, i in walls if not is_partition(p)]]
    labels += ['partition', 'facade']
    links = -grid.laplacian.copy()
    np.fill_diagonal(links, 0)
    # [a,b] is heat arriving in group a from group b, at the solved field.
    flux = np.zeros((len(masks), len(masks)))
    for a, receiving in enumerate(masks):
        for b, sending in enumerate(masks):
            if a != b:
                flux[a, b] = np.sum(links[np.ix_(receiving, sending)]
                    * (updated[sending][None, :] - updated[receiving][:, None]))
    group_residual = [float(storage[m].sum() - source[m].sum()
                            - outside[m].sum() - flux[a].sum()) for a, m in enumerate(masks)]
    return dict(total_balance_residual_W=float(storage.sum() - source.sum() - outside.sum()),
                maximum_node_balance_residual_W=float(np.max(np.abs(residual))),
                zone_air_balance_residual_W=(grid.sensed.T @ residual
                    * np.count_nonzero(grid.sensed, axis=0)).tolist(),
                energy_group_labels=labels, group_exchange_W=flux.tolist(),
                group_balance_residual_W=group_residual,
                air_stream_power_W=air_power.tolist())


def report():
    grid = fixture()
    previous = np.full(len(grid.capacity), 295.)
    # Transparent nonuniform room history with identical zone mean.
    patterned = previous.copy()
    for mask in grid.rooms:
        members = [grid.index[p] for p in zip(*np.where(mask))]
        patterned[members] += np.linspace(-2., 2., len(members))
    basis = reduced_basis(grid)
    rows = []
    for name, start in [('uniform', previous), ('nonuniform', patterned)]:
        full = solve_reference(grid, start)
        source = solve_source(grid, start)
        reduced = solve_reference(grid, start, basis=basis)
        rows.append(dict(initial_state=name,
            source_reference_maximum_error_K=float(np.max(np.abs(source - full))),
            reference_zone_temperatures_K=(grid.sensed.T @ full).tolist(),
            reduced_zone_temperatures_K=(grid.sensed.T @ reduced).tolist(),
            reduced_zone_error_K=(grid.sensed.T @ (reduced - full)).tolist(),
            full_energy=energy_audit(grid, start, full),
            reduced_energy=energy_audit(grid, start, reduced)))
    return dict(protocol='Synthetic single-step algebra fixtures only; not a building rollout or score',
        full_states=len(previous), reduced_states=basis.shape[1],
        reduction='Piecewise-constant room air/slab; all wall CVs retained; no fitting',
        fixture_constants='All explicitly defined in tools/thermal_grid_reference.py:fixture', rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error('Refusing to overwrite a report')
    result = report()
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.output:
        with args.output.open('x') as file:
            json.dump(result, file, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
