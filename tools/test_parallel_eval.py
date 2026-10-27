"""Exact screen metrics without running a simulator."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.parallel_eval import one_job, read_exact_return


class ParallelEvalTest(unittest.TestCase):
    def test_exact_return_and_partial_run_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'building/weather/policy/summary.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(total_reward=-.87654321, total_steps=288)))
            self.assertEqual(read_exact_return(directory, 'building', 'weather', 'policy', 288), -.87654321)
            with self.assertRaises(ValueError):
                read_exact_return(directory, 'building', 'weather', 'policy', 287)

    def test_stdout_is_not_a_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('tools.parallel_eval.subprocess.run') as run:
                run.return_value.returncode = 0
                run.return_value.stdout = ' policy: reward=-0.88'
                _, reward, note = one_job(('policy', 'building', 'weather', 288, 10, False, directory))
            self.assertIsNone(reward)
            self.assertIn('invalid summary', note)
            self.assertIn('--results-dir', run.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
