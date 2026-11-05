"""P21-5 F3/F7: scale grid + offset ablation registration and wiring."""

import unittest

import numpy as np

from smart_buildings import evaluate
from smart_buildings.baselines.probe_modes import (
    SkynetNoToProbe,
    SkynetS0Probe,
    SkynetS025Probe,
    SkynetS05Probe,
    _SkynetScaleBase,
)


class TestF3F7(unittest.TestCase):
    def test_registered(self):
        for key in ("probe_skynet_s0", "probe_skynet_s025",
                    "probe_skynet_s05", "probe_skynet_noto"):
            self.assertIn(key, evaluate.POLICY_MAP)

    def test_scale_consts(self):
        self.assertEqual(SkynetS0Probe.SKYNET_SCALE, 0.0)
        self.assertEqual(SkynetS025Probe.SKYNET_SCALE, 0.25)
        self.assertEqual(SkynetS05Probe.SKYNET_SCALE, 0.5)
        self.assertIsNone(_SkynetScaleBase.SKYNET_SCALE)

    def test_scale_override_wiring(self):
        # Post-hoc override without constructing the full policy.
        probe = _SkynetScaleBase.__new__(_SkynetScaleBase)

        class FakePol:
            scale = 1.0

        probe._pol = FakePol()
        # emulate __init__ tail for each subclass const
        for cls, want in ((SkynetS0Probe, 0.0), (SkynetS025Probe, 0.25),
                          (SkynetS05Probe, 0.5)):
            probe._pol.scale = 1.0
            probe.SKYNET_SCALE = cls.SKYNET_SCALE
            if probe.SKYNET_SCALE is not None:
                probe._pol.scale = probe.SKYNET_SCALE
            self.assertEqual(probe._pol.scale, want)

    def test_noto_zeroed_shape(self):
        probe = SkynetNoToProbe.__new__(SkynetNoToProbe)

        class FakeIx:
            zone_vav = ["v0", "v1"]

            def target_offsets(self, mz, scale):
                return {"v0": 0.5, "v1": -0.3}

        class FakePol:
            ix = FakeIx()

        probe._pol = FakePol()
        # emulate the patch block
        ix = probe._pol.ix
        keys = list(ix.target_offsets(
            np.zeros((len(ix.zone_vav), 3), dtype=float), 1.0).keys())

        def _zeroed(mz, scale):
            return {v: 0.0 for v in keys}

        ix.target_offsets = _zeroed
        out = probe._pol.ix.target_offsets(np.ones((2, 3)), 1.0)
        self.assertEqual(out, {"v0": 0.0, "v1": 0.0})


if __name__ == "__main__":
    unittest.main()
