import os
import subprocess
import sys
from pathlib import Path


def test_import_database_does_not_create_sqlite_file(tmp_path):
    database_path = tmp_path / "not_created.sqlite3"

    env = os.environ.copy()
    env["CAUSALMEM_DB"] = str(database_path)
    env["MEM0_PATH"] = str(tmp_path / "mem0")
    env["MEM0_DIR"] = str(tmp_path / "mem0-config")

    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "-c", "import database"],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    assert not database_path.exists(), (
        "Importing database unexpectedly created the SQLite file."
    )