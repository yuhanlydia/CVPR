"""SSH argument construction using an engineering transport double; no remote connection."""
import json
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_remote.sh"


class RemoteArgumentTests(unittest.TestCase):
    def test_project_path_is_quoted_as_one_literal_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "ssh"
            fake.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            fake.chmod(0o755)
            env = dict(os.environ, PATH=directory + os.pathsep + os.environ["PATH"],
                       GPU_HOST="engineering-host", GPU_PROJECT_DIR="/fixture/O'Brien $literal")
            run = subprocess.run(["bash", str(SCRIPT), "python3 scripts/inspect_host.py"],
                                 env=env, text=True, capture_output=True, check=True)
            argv = json.loads(run.stdout)
            self.assertEqual(argv[:2], ["--", "engineering-host"])
            directory_command = argv[2].split(" && ", 1)[0]
            self.assertEqual(shlex.split(directory_command), ["cd", "--", env["GPU_PROJECT_DIR"]])

    def test_extra_command_arguments_are_rejected(self):
        env = dict(os.environ, GPU_HOST="engineering-host", GPU_PROJECT_DIR="/fixture")
        run = subprocess.run(["bash", str(SCRIPT), "first", "silently-ignored-before"],
                             env=env, text=True, capture_output=True)
        self.assertEqual(run.returncode, 2)
        self.assertIn("Usage:", run.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
