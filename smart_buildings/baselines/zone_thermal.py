"""Stable lumped thermal integration and bounded, prefix-only identification.

Equipment flows/powers are inputs, not fitted valve-to-temperature gains.
Capacities are J/K, conductances W/K, powers W and time seconds. This module
does not implement control or infer that fitted parameters are identifiable.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import least_squares

from smart_buildings.baselines.equipment_balance import AIR_CAPACITY, ZoneThermalParameters


def _advance(p, air, mass, outdoor, supply, flow, reheat, internal, seconds):
    """Backward Euler M-matrix solve; air exchange remains coupled to T_next."""
    ga = AIR_CAPACITY * flow
    a = 1 + seconds * (p.outdoor_conductance + p.mass_conductance + ga) / p.air_capacity
    b = seconds * p.mass_conductance / p.air_capacity
    c = seconds * p.mass_conductance / p.mass_capacity
    d = 1 + seconds * (p.mass_conductance + p.mass_outdoor_conductance) / p.mass_capacity
    rhs_air = air + seconds * (p.outdoor_conductance * outdoor + ga * supply
                               + reheat + internal) / p.air_capacity
    rhs_mass = mass + seconds * p.mass_outdoor_conductance * outdoor / p.mass_capacity
    # Expanded determinant avoids cancellation of b*c for strong coupling.
    exterior_air = seconds * (p.outdoor_conductance + ga) / p.air_capacity
    exterior_mass = seconds * p.mass_outdoor_conductance / p.mass_capacity
    determinant = (1 + exterior_air) * (1 + exterior_mass) + b * (1 + exterior_mass) + c * (1 + exterior_air)
    return ((d * rhs_air + b * rhs_mass) / determinant,
            (c * rhs_air + a * rhs_mass) / determinant)


def advance_zone(parameters, air, mass, *, outdoor, supply, flow,
                 reheat=0.0, internal=0.0, seconds=300.0):
    if not np.isfinite([air, mass, outdoor, supply, flow, reheat, internal, seconds]).all():
        raise ValueError('Thermal inputs must be finite')
    if flow < 0 or seconds < 0:
        raise ValueError('Flow and duration must be nonnegative')
    result = _advance(parameters, air, mass, outdoor, supply, flow, reheat, internal, seconds)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite thermal integration result')
    return result


def condition_mass(parameters, previous_mass, measured_air, outdoor, seconds=300.0):
    """Wall-state update conditioned on an ALREADY received air measurement."""
    p = parameters
    return (previous_mass + seconds * (p.mass_conductance * measured_air
            + p.mass_outdoor_conductance * outdoor) / p.mass_capacity) / (
                1 + seconds * (p.mass_conductance + p.mass_outdoor_conductance) / p.mass_capacity)


def condition_mass_kalman(parameters, previous_air, previous_mass, mass_variance,
                          measured_air, *, outdoor, supply, flow, reheat=0.0,
                          internal=0.0, seconds=300.0, air_process_std=0.05,
                          mass_process_std=0.01):
    """Condition an uncertain mass node on an exact received air measurement.

    This is the scalar form of a two-state Kalman update with air pinned to
    its observed value between updates. Q represents model mismatch, not fitted
    validation errors. No temperature target is used until AFTER propagation.
    """
    if (not np.isfinite([previous_air, previous_mass, mass_variance, measured_air,
                         air_process_std, mass_process_std]).all()
            or mass_variance < 0 or air_process_std <= 0 or mass_process_std < 0):
        raise ValueError('Invalid state variance or process-noise assumptions')
    predicted_air, predicted_mass = advance_zone(
        parameters, previous_air, previous_mass, outdoor=outdoor, supply=supply,
        flow=flow, reheat=reheat, internal=internal, seconds=seconds)
    p = parameters
    exterior_air = seconds * (p.outdoor_conductance + AIR_CAPACITY * flow) / p.air_capacity
    exterior_mass = seconds * p.mass_outdoor_conductance / p.mass_capacity
    b = seconds * p.mass_conductance / p.air_capacity
    c = seconds * p.mass_conductance / p.mass_capacity
    determinant = (1 + exterior_air) * (1 + exterior_mass) + b * (1 + exterior_mass) + c * (1 + exterior_air)
    f01, f11 = b / determinant, (1 + exterior_air + b) / determinant
    innovation_variance = f01 ** 2 * mass_variance + air_process_std ** 2
    gain = f01 * f11 * mass_variance / innovation_variance
    mass = predicted_mass + gain * (measured_air - predicted_air)
    # Joseph-style expression avoids subtracting two nearly equal variances.
    variance = ((f11 - gain * f01) ** 2 * mass_variance
                + (gain * air_process_std) ** 2 + mass_process_std ** 2)
    if not np.isfinite([mass, variance]).all():
        raise ValueError('Nonfinite mass-state estimate')
    return mass, variance


@dataclass(frozen=True)
class ThermalSeries:
    start: np.ndarray
    end: np.ndarray
    outdoor: np.ndarray
    supply: np.ndarray
    flow: np.ndarray
    reheat: np.ndarray
    comfort: np.ndarray

    def __post_init__(self):
        arrays = [np.asarray(value, dtype=float) for value in vars(self).values()]
        if (not arrays or arrays[0].ndim != 1 or len(arrays[0]) == 0
                or any(array.shape != arrays[0].shape or not np.isfinite(array).all() for array in arrays)):
            raise ValueError('Thermal series require matching nonempty finite vectors')
        if np.any(arrays[4] < 0) or np.any((arrays[6] < 0) | (arrays[6] > 1)):
            raise ValueError('Invalid flow or comfort fraction')
        for name, value in zip(vars(self), arrays):
            copied = value.copy()
            copied.flags.writeable = False
            object.__setattr__(self, name, copied)

    def prefix(self, count):
        if not 1 <= count <= len(self.start):
            raise ValueError('Invalid training prefix length')
        return ThermalSeries(**{name: value[:count] for name, value in vars(self).items()})


@dataclass(frozen=True)
class ThermalFit:
    parameters: ZoneThermalParameters
    internal_comfort_power: float
    initial_mass: float
    standby_fraction: float
    two_state: bool
    diagnostics: dict

    def internal_power(self, comfort):
        return self.internal_comfort_power * (self.standby_fraction + (1 - self.standby_fraction) * comfort)


def fit_zone(series, *, two_state=True, area_scale=1.0, seconds=300.0,
             mass_capacity_ratio=20.0, standby_fraction=0.2, max_evaluations=40,
             mass_observer='boundary', passive_multistep=False):
    """Fit C_air, U_out, internal load; optionally U_mass and initial mass offset.

    C_mass/C_air and standby fraction remain explicit fixed priors: a single
    six-hour prefix cannot identify every slow parameter. Only the supplied
    prefix is accepted; no scenario IDs or validation targets are read.
    """
    if (not np.isfinite([area_scale, seconds, mass_capacity_ratio, standby_fraction]).all()
            or area_scale <= 0 or seconds <= 0 or mass_capacity_ratio <= 0
            or not 0 <= standby_fraction <= 1 or max_evaluations < 1):
        raise ValueError('Invalid fitting priors or budget')
    if mass_observer not in ('boundary', 'kalman'):
        raise ValueError('Unknown mass-state observer')
    if passive_multistep and mass_observer != 'boundary':
        raise ValueError('Passive multistep fitting requires boundary mass updates')
    if len(series.start) < 8:
        raise ValueError('At least eight observed transitions are required')
    capacity_prior = float(np.clip(3e6 * area_scale, 1e5, 1e9))
    outside_prior = float(np.clip(20 * math.sqrt(area_scale), 1e-3, 1e5))
    mass_prior = float(np.clip(capacity_prior / 7200, 1e-3, 1e5))
    load_prior = float(np.clip(1000 * area_scale, 0, 50000))
    initial = np.array([math.log(capacity_prior), math.log(outside_prior), load_prior / 1000])
    lower = [math.log(1e5), math.log(1e-3), 0]
    upper = [math.log(1e9), math.log(1e5), 50]
    if two_state:
        initial = np.r_[initial, math.log(mass_prior), 0.0]
        offset_bound = 3.0 if passive_multistep else 15.0
        lower += [math.log(1e-3), -offset_bound]
        upper += [math.log(1e5), offset_bound]

    def unpack(vector):
        capacity = math.exp(vector[0])
        p = ZoneThermalParameters(capacity, mass_capacity_ratio * capacity,
                                  math.exp(vector[1]), math.exp(vector[3]) if two_state else 0, 0)
        mass = float(series.start[0] + (vector[4] if two_state else 0))
        return p, 1000 * vector[2], mass

    def residuals(vector, regularize=True):
        p, load, mass = unpack(vector)
        variance = 4.0
        errors = []
        mass_path = []
        for start, end, outdoor, supply, flow, reheat, comfort in zip(*vars(series).values()):
            mass_path.append(mass)
            power = load * (standby_fraction + (1 - standby_fraction) * comfort)
            predicted, _ = _advance(p, start, mass, outdoor, supply, flow, reheat, power, seconds)
            errors.append(predicted - end)
            # The target enters the state estimator only AFTER the prediction.
            if mass_observer == 'kalman':
                mass, variance = condition_mass_kalman(
                    p, start, mass, variance, end, outdoor=outdoor,
                    supply=supply, flow=flow, reheat=reheat, internal=power, seconds=seconds)
            else:
                mass = condition_mass(p, mass, end, outdoor, seconds)
        if passive_multistep:
            # All targets/forcing are inside the supplied training prefix.
            # Each horizon has equal aggregate weight. State at each origin
            # uses received air only; rollout mass never assimilates targets.
            horizon_errors = []
            for horizon in (1, 2, 6, 12):
                values = []
                for origin in range(len(series.start) - horizon + 1):
                    air, latent = series.start[origin], mass_path[origin]
                    for index in range(origin, origin + horizon):
                        power = load * (standby_fraction + (1 - standby_fraction) * series.comfort[index])
                        air, latent = _advance(p, air, latent, series.outdoor[index],
                            series.supply[index], series.flow[index], series.reheat[index], power, seconds)
                    values.append((air - series.end[origin + horizon - 1]) / horizon)
                if values:
                    horizon_errors.extend(np.asarray(values) * math.sqrt(len(series.start) / len(values)))
            errors = horizon_errors
        if not regularize:
            return np.array(errors)
        scales = np.array([math.log(10), math.log(10), max(1.0, load_prior / 1000)])
        if two_state:
            scales = np.r_[scales, math.log(10), 5.0]
        return np.r_[errors, 0.05 * (vector - initial) / scales]

    result = least_squares(residuals, initial, bounds=(lower, upper),
                           max_nfev=max_evaluations, ftol=1e-6, xtol=1e-6, gtol=1e-6)
    p, load, mass = unpack(result.x)
    data_error = residuals(result.x, regularize=False)
    lower, upper = np.array(lower), np.array(upper)
    relative = (result.x - lower) / (upper - lower)
    diagnostics = dict(mass_observer=mass_observer,
                       fitting_objective='passive_multistep' if passive_multistep else 'one_step',
                       initial_mass_offset_bound_K=(3.0 if passive_multistep else 15.0) if two_state else 0.0,
                       training_error_definition='horizon-normalized residuals' if passive_multistep else 'one-step residuals',
                       training_transitions=len(series.start), solver_success=bool(result.success),
                       solver_status=int(result.status), evaluations=int(result.nfev),
                       training_mae_K=float(np.mean(np.abs(data_error))),
                       parameters_at_bounds=int(np.count_nonzero((relative < 1e-3) | (relative > .999))),
                       active_transitions=int(np.count_nonzero(series.flow > 1e-6)),
                       mass_capacity_ratio=mass_capacity_ratio,
                       identified_all_parameters=False)
    fit = ThermalFit(p, float(load), mass, standby_fraction, two_state, diagnostics)
    diagnostics['prefix_balance'] = prefix_balance_diagnostics(fit, series, seconds=seconds)
    # Data-only sensitivities: exclude the prior residuals, which can mask
    # degeneracy. Coordinates are log(C), log(U), load/1000 W, log(G), offset/K.
    columns = []
    for index in range(len(result.x)):
        delta = np.zeros_like(result.x)
        delta[index] = 1e-4
        columns.append((residuals(result.x + delta, False)
                        - residuals(result.x - delta, False)) / 2e-4)
    singular = np.linalg.svd(np.array(columns).T, compute_uv=False)
    diagnostics['data_sensitivity'] = dict(
        coordinates=['log_C_air', 'log_U_out', 'load_kW']
        + (['log_G_mass', 'initial_mass_offset_K'] if two_state else []),
        singular_values=singular.tolist(),
        smallest_to_largest_ratio=float(singular[-1] / singular[0]) if singular[0] > 0 else 0.0,
        interpretation='Local data-only sensitivity in stated coordinates, not an identifiability proof')
    return fit


def prefix_balance_diagnostics(fit, series, *, seconds=300.0):
    """Audit observed air balance with a passive mass path, never Kalman heat.

    Uses only the supplied prefix. Required mass is the next-step temperature
    that would close the backward-Euler air balance exactly; it is diagnostic,
    not a state estimate or an input to fitting/control.
    """
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError('Duration must be finite and positive')
    p = fit.parameters
    mass = fit.initial_mass
    low = high = mass
    residuals, violations, required = [], [], []
    for start, end, outdoor, supply, flow, reheat, comfort in zip(*vars(series).values()):
        low, high = min(low, start, end), max(high, start, end)
        mass = condition_mass(p, mass, end, outdoor, seconds)
        storage = p.air_capacity * (end - start) / seconds
        known = (p.outdoor_conductance * (outdoor - end)
                 + AIR_CAPACITY * flow * (supply - end)
                 + reheat + fit.internal_power(comfort))
        residuals.append(storage - known - p.mass_conductance * (mass - end))
        if p.mass_conductance > 0:
            needed = end + (storage - known) / p.mass_conductance
            required.append(needed)
            violations.append(max(low - needed, needed - high, 0.0))
    return dict(transitions=len(residuals),
                passive_balance_mae_W=float(np.mean(np.abs(residuals))),
                passive_balance_mean_W=float(np.mean(residuals)),
                passive_balance_p95_W=float(np.percentile(np.abs(residuals), 95)),
                required_mass_range_K=[float(min(required)), float(max(required))] if required else None,
                required_mass_outside_envelope=int(np.count_nonzero(np.array(violations) > 1e-6)),
                maximum_required_mass_envelope_violation_K=float(max(violations)) if violations else None)


def filtered_mass_path(fit, series, *, seconds=300.0):
    """Index 0 is the first start state; index i+1 uses targets only through i."""
    masses = [fit.initial_mass]
    for end, outdoor in zip(series.end, series.outdoor):
        masses.append(condition_mass(fit.parameters, masses[-1], end, outdoor, seconds))
    return np.array(masses)


@dataclass(frozen=True)
class ThermalForecast:
    temperatures: tuple
    mass_temperatures: tuple
    uncertain_damper_floors: tuple
    uncertain_pressure_floors: tuple


def forecast_path(observer, fits, mass_states, commands, steps, *,
                  pending_comfort, current_comfort, seconds=300.0):
    """Propagate pending physics first, then the supplied candidate command path.

    Only origin observations are available. Future weather and comfort flags
    are held; water is rolled using previously observed rates. Recorded future
    commands may be supplied for OFFLINE conditional-dynamics validation, never
    described as an unconditional online forecast.
    """
    if observer.pending is None or steps < 1 or len(commands) < steps - 1:
        raise ValueError('Pending physics and a complete command path are required')
    if set(fits) != set(observer.rooms) or set(mass_states) != set(observer.rooms):
        raise ValueError('Fits and mass estimates must cover all zones')
    temperatures = {room: observer.frame['zones'][room]['temperature'] for room in observer.rooms}
    masses = dict(mass_states)
    water = observer.frame['water']
    outdoor = {ahu: observer.frame['handlers'][ahu]['outside_air_temperature_sensor']
               for ahu in observer.ahu_ids}
    room_ahu = {room: ahu for ahu, rooms in observer.handlers.items() for room in rooms}
    path, mass_path = [], []
    uncertain_dampers, uncertain_pressures = set(), set()
    for step in range(steps):
        if step == 0:
            forcing, comfort = observer.pending, pending_comfort
        else:
            command = commands[step - 1]
            forcing = observer.project_action(*command, zone_temperatures=temperatures,
                                              outdoor_temperatures=outdoor, water_temperature=water)
            comfort = current_comfort
        uncertain_dampers.update(forcing.uncertain_damper_floors)
        uncertain_pressures.update(forcing.uncertain_pressure_floors)
        for room in observer.rooms:
            fit, duty = fits[room], forcing.zones[room]
            temperatures[room], masses[room] = advance_zone(
                fit.parameters, temperatures[room], masses[room],
                outdoor=forcing.outdoor_temperature[room_ahu[room]],
                supply=duty.supply_temperature, flow=duty.air_flow,
                reheat=duty.reheat_power, internal=fit.internal_power(comfort), seconds=seconds)
        if step > 0:
            water = observer.project_water_sensor(commands[step - 1][0], start_temperature=water)
        path.append(temperatures.copy())
        mass_path.append(masses.copy())
    return ThermalForecast(tuple(path), tuple(mass_path), tuple(sorted(uncertain_dampers)),
                           tuple(sorted(uncertain_pressures)))
