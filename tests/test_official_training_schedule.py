from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class OfficialTrainingScheduleTests(unittest.TestCase):
    def test_segment_stops_exactly_at_next_validation_step(self):
        from official_training_schedule import training_segment_update_limit

        self.assertEqual(
            training_segment_update_limit(
                global_step=900,
                max_iterations=30000,
                val_interval_iterations=1000,
                next_val_step=1000,
            ),
            100,
        )
        self.assertEqual(
            training_segment_update_limit(
                global_step=29950,
                max_iterations=30000,
                val_interval_iterations=1000,
                next_val_step=30000,
            ),
            50,
        )

    def test_epoch_recipe_has_no_artificial_segment_limit(self):
        from official_training_schedule import training_segment_update_limit

        self.assertIsNone(
            training_segment_update_limit(
                global_step=0,
                max_iterations=None,
                val_interval_iterations=None,
                next_val_step=0,
            )
        )

    def test_runner_uses_schedule_helper_and_unbounded_iteration_loop(self):
        runner = (ROOT / "src/official_baseline_common.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("training_segment_update_limit(", runner)
        self.assertIn("while max_iterations is not None or epoch < int(args.epochs):", runner)


if __name__ == "__main__":
    unittest.main()
