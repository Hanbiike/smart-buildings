"""P20-6 R2: PtrLoopMixin placement, hook defaults, sub-lever liveness."""

import unittest

from smart_buildings import evaluate
from smart_buildings.baselines.occ_dbh import DeadbandHeatMixin
from smart_buildings.baselines.occ_gated7 import OccGated7Policy
from smart_buildings.baselines.occ_ptr1 import (
    OccPtrA1Policy,
    OccPtrB1Policy,
    OccPtrC1Policy,
    Ptr1Base,
    PtrLoopMixin,
)


def _bare(cls):
    return cls.__new__(cls)


class TestPtr1Wiring(unittest.TestCase):
    def test_registered(self):
        for key in ("occ_ptr0", "occ_ptra1", "occ_ptrb1", "occ_ptrc1"):
            self.assertIn(key, evaluate.POLICY_MAP)

    def test_mixin_sits_between_dbh_and_gated7(self):
        mro = list(Ptr1Base.__mro__)
        self.assertLess(mro.index(DeadbandHeatMixin), mro.index(PtrLoopMixin))
        self.assertLess(mro.index(PtrLoopMixin), mro.index(OccGated7Policy))

    def test_hook_defaults_reproduce_fixed_loop(self):
        b = _bare(Ptr1Base)
        self.assertEqual(b._ptr_up_eff(5), (b.PTR_UP, b.PTR_PERSIST))
        self.assertTrue(b._ptr_starved_fire(5, 1))
        self.assertFalse(b._ptr_starved_fire(5, 0))
        self.assertEqual(b._ptr_db_eff(0, 1.25), 1.25)

    def test_sub_a_normalizes_by_ahu_size(self):
        a = _bare(OccPtrA1Policy)
        self.assertEqual(a._ptr_up_eff(1), (a.PTR_UP, a.PTR_PERSIST))
        up, persist = a._ptr_up_eff(4)
        self.assertAlmostEqual(up, a.PTR_UP / 2.0)
        self.assertEqual(persist, a.PTR_PERSIST + 3)

    def test_sub_b_rogue_exclusion(self):
        b = _bare(OccPtrB1Policy)
        self.assertFalse(b._ptr_starved_fire(4, 1))
        self.assertTrue(b._ptr_starved_fire(4, 2))
        self.assertTrue(b._ptr_starved_fire(2, 1))

    def test_sub_c_ema_db(self):
        c = _bare(OccPtrC1Policy)
        c._ptr_req_ema = [0.0, 0.5]
        self.assertEqual(c._ptr_db_eff(0, 1.0), 1.0)
        self.assertAlmostEqual(c._ptr_db_eff(1, 1.0), 1.5)


if __name__ == "__main__":
    unittest.main()
