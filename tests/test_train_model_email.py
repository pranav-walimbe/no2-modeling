"""Verify model-training email attachments."""

import unittest
from pathlib import Path


class TrainingEmailTest(unittest.TestCase):
    def test_email_attaches_comparison_then_loss_plot(self) -> None:
        script = (Path(__file__).parents[1] / "scripts" / "slurm" / "train_model.sh").read_text()
        mail_command = script.split('"mailx ', maxsplit=1)[1]

        self.assertIn("-a '${comparison_plot}' -a '${loss_plot}'", mail_command)
        self.assertNotIn("results_file", mail_command)


if __name__ == "__main__":
    unittest.main()
