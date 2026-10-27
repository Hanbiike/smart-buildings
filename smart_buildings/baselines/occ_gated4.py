"""Occupancy-gated gen-3.1, round 4: round-3 without the hot-water floor.

Round-2/3 full episodes showed the OAT-scheduled hot-water floor adds gas
without reducing airflow. The VAV coil explains why: discharge air is capped
at 308.15 K, so once the coil hits the cap, hotter water is clipped and
delivers no extra heat per unit air -- the PI equilibrium (and the
ventilation load) does not move, while the hotter loop pays more tank
dissipation and cycling. The floor is removed; the water reset underneath is
left to keep the loop just hot enough.

  OccGated4Policy: round-3 (latch + bounded floor + prestart-only rates +
    idle pump-off) with the hot-water floor disabled.
"""

from smart_buildings.baselines.occ_gated3 import OccGated3Policy


class OccGated4Policy(OccGated3Policy):
    """Round-3 with the hot-water floor disabled."""

    KEY = "occ_gated4"

    def _apply_water_floor(self, obs, action) -> None:
        return None
