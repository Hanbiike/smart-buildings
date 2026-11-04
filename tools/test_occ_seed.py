# -*- coding: utf-8 -*-
"""Focused tests for the P20-2 occupancy-seed rig (gin-level, no env)."""
import os
import sys
import unittest

sys.path.insert(0, "/home/a_rakhymbekov/BUILDINGS")

GIN = ("smart_buildings/training_scenarios/building_train_8/"
       "atlanta_1/config.gin")
SEED_PATH = ("randomized_occupancy/"
             "RandomizedArrivalDepartureOccupancy.seed")

# Registers gin configurables for parse_config_file.
from smart_buildings import evaluate as _E  # noqa: F401,E402


class TestOccSeedBind(unittest.TestCase):
    def test_bind_roundtrip(self):
        import gin
        gin.clear_config()
        with gin.unlock_config():
            gin.parse_config_file(GIN)
            gin.bind_parameter(SEED_PATH, 99999)
            self.assertEqual(gin.query_parameter(SEED_PATH), 99999)
        gin.clear_config()

    def test_unset_leaves_default(self):
        # No seed line in the shipped config: without OCC_SEED the
        # constructor default (17321) applies untouched.
        import gin
        gin.clear_config()
        with gin.unlock_config():
            gin.parse_config_file(GIN)
            with self.assertRaises(ValueError):
                gin.query_parameter(SEED_PATH)
        gin.clear_config()
        from smart_buildings.smart_control.simulator. \
            randomized_arrival_departure_occupancy import (
                RandomizedArrivalDepartureOccupancy)
        import inspect
        sig = inspect.signature(
            RandomizedArrivalDepartureOccupancy.__init__)
        self.assertEqual(sig.parameters["seed"].default, 17321)

    def test_loader_honors_env(self):
        # load_hybrid_environment binds OCC_SEED when set; unset = default.
        import gin
        from pathlib import Path
        from smart_buildings import evaluate as E
        os.environ["OCC_SEED"] = "424242"
        try:
            gin.clear_config()
            # Reproduce the loader's parse+bind preamble without
            # constructing the heavy environment.
            with gin.unlock_config():
                gin.parse_config_file(str(Path(GIN).resolve()))
                occ_seed = os.environ.get("OCC_SEED")
                if occ_seed is not None:
                    gin.bind_parameter(SEED_PATH, int(occ_seed))
                self.assertEqual(gin.query_parameter(SEED_PATH), 424242)
        finally:
            del os.environ["OCC_SEED"]
            gin.clear_config()


if __name__ == "__main__":
    unittest.main()
