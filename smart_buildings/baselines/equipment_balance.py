"""Pure equipment energy balances; no controller, fitting, prices or hidden config.

All temperatures are kelvin, flows m3/s, powers watts, pressures Pa for air
and bar for water. Equipment parameters must be supplied explicitly. AHU
setpoints are not discharge measurements. No numerical integration or sensor
timing assumption is made here.
"""

from dataclasses import dataclass
import math

import numpy as np

AIR_CAPACITY = 1.2 * 1006.0  # volumetric heat capacity, J/(m3 K)
WATER_CAPACITY = 1000.0 * 4180.0


def _nonnegative(**values):
    for name, value in values.items():
        array = np.asarray(value, dtype=float)
        if not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError(f'{name} must be finite and nonnegative')


def _temperatures(*values):
    if not np.isfinite(values).all():
        raise ValueError('Temperatures must be finite')


def air_flows(dampers, design_flows, pressure_pa, design_pressure_pa,
              handler_capacity, *, running):
    """Pressure-driven zone flows, reduced together by a handler flow cap.

    Dampers are effective openings AFTER actuator floors, not raw commands.
    """
    dampers, design_flows = np.asarray(dampers, float), np.asarray(design_flows, float)
    _nonnegative(dampers=dampers, design_flows=design_flows, pressure=pressure_pa,
                 design_pressure=design_pressure_pa, capacity=handler_capacity)
    if dampers.ndim != 1 or dampers.shape != design_flows.shape or np.any(dampers > 1):
        raise ValueError('Matching one-dimensional arrays and dampers in [0,1] required')
    if design_pressure_pa <= 0:
        raise ValueError('Design pressure must be positive')
    flows = dampers * design_flows * math.sqrt(pressure_pa / design_pressure_pa)
    if not running:
        return np.zeros_like(flows)
    total = float(flows.sum())
    return flows * (min(1.0, handler_capacity / total) if total else 1.0)


@dataclass(frozen=True)
class AirHandlerBalance:
    supply_temperature: float
    cooling_power: float
    fan_power: float


def air_handler(return_temperature, outdoor_temperature, recirculation,
                cooling_setpoint, flow, pressure_pa, efficiency, *, running):
    _temperatures(return_temperature, outdoor_temperature, cooling_setpoint)
    _nonnegative(flow=flow, pressure=pressure_pa)
    if not 0 <= recirculation <= 1 or not 0 < efficiency <= 1:
        raise ValueError('Invalid recirculation fraction or fan efficiency')
    mixed = recirculation * return_temperature + (1 - recirculation) * outdoor_temperature
    if not running:
        if flow != 0:
            raise ValueError('An off handler cannot deliver nonzero airflow')
        return AirHandlerBalance(mixed, 0.0, 0.0)
    supply = min(mixed, cooling_setpoint)
    return AirHandlerBalance(supply, AIR_CAPACITY * flow * (mixed - supply),
                             flow * pressure_pa / efficiency)


def water_flows(valves, valve_kv, pressure_bar, header_resistance, *, running):
    valves, valve_kv = np.asarray(valves, float), np.asarray(valve_kv, float)
    _nonnegative(valves=valves, kv=valve_kv, pressure=pressure_bar,
                 resistance=header_resistance)
    if valves.ndim != 1 or valves.shape != valve_kv.shape or np.any(valves > 1):
        raise ValueError('Matching one-dimensional arrays and valves in [0,1] required')
    effective = valves * valve_kv
    if not running:
        return np.zeros_like(effective)
    # Kv relation yields m3/hour; all downstream balances require m3/second.
    return effective * math.sqrt(pressure_bar / (
        1 + float(effective.sum()) ** 2 * header_resistance)) / 3600


@dataclass(frozen=True)
class CoilBalance:
    discharge_temperature: float
    return_water_temperature: float
    heat_to_air: float


def reheat_coil(supply_temperature, water_temperature, air_flow, water_flow,
                max_discharge_temperature):
    _temperatures(supply_temperature, water_temperature, max_discharge_temperature)
    _nonnegative(air_flow=air_flow, water_flow=water_flow)
    air_capacity = AIR_CAPACITY * air_flow
    water_capacity = WATER_CAPACITY * water_flow
    if air_capacity == 0 or water_capacity == 0:
        return CoilBalance(supply_temperature, water_temperature, 0.0)
    gap = water_temperature - supply_temperature
    power = min(air_capacity, water_capacity) * gap
    if gap > 0:
        power = min(power, air_capacity * max(0, max_discharge_temperature - supply_temperature))
    return CoilBalance(supply_temperature + power / air_capacity,
                       water_temperature - power / water_capacity, power)


@dataclass(frozen=True)
class ZoneThermalParameters:
    air_capacity: float  # effective J/K, not automatically just air mass
    mass_capacity: float
    outdoor_conductance: float  # W/K; direct path, e.g. infiltration
    mass_conductance: float
    mass_outdoor_conductance: float

    def __post_init__(self):
        _nonnegative(**vars(self))
        if self.air_capacity <= 0 or self.mass_capacity <= 0:
            raise ValueError('Thermal capacities must be positive')


def zone_derivatives(parameters, temperature, mass_temperature, outdoor_temperature,
                     supply_temperature, air_flow, reheat_power=0.0,
                     internal_power=0.0):
    """Return air/mass derivatives in K/s; use PRE-reheat supply temperature.

    Passing post-coil discharge and also reheat_power would double-count heat.
    The exchange with the mass node cancels in total stored-energy change.
    """
    _temperatures(temperature, mass_temperature, outdoor_temperature, supply_temperature,
                  reheat_power, internal_power)
    _nonnegative(air_flow=air_flow)
    p = parameters
    exchange = p.mass_conductance * (mass_temperature - temperature)
    air_power = (p.outdoor_conductance * (outdoor_temperature - temperature)
                 + exchange + AIR_CAPACITY * air_flow * (supply_temperature - temperature)
                 + reheat_power + internal_power)
    mass_power = (-exchange + p.mass_outdoor_conductance *
                  (outdoor_temperature - mass_temperature))
    return air_power / p.air_capacity, mass_power / p.mass_capacity
