from __future__ import annotations

from pathlib import Path

from scripts.publication import check_publication_hygiene as hygiene


def test_publication_hygiene_has_no_banned_names_paths_or_literal_secrets():
    violations = hygiene.collect_violations()
    assert violations == []


def test_filename_detector_flags_publication_specific_cases(monkeypatch, tmp_path):
    monkeypatch.setattr(hygiene, "ROOT", tmp_path)
    names = [
        "experiment_" + "rebut" + "tal.py",
        "Open" + "Review_notes.md",
        "table_" + "IC" + "ML.py",
        "author" + " response.md",
        "sync_" + "Mere" + "nova.py",
        "--" + "api" + "-key.txt",
        "bare_" + "api" + "-key.txt",
        "ssh_" + "ro" + "ot@host.sh",
        "host_" + "203." + "0.113." + "9.sh",
        str(Path("home") / "hyunjin" / "config.py"),
    ]

    violations = []
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        violations.extend(hygiene.filename_violations(path))

    assert len(violations) == len(names)


def test_text_detector_flags_cli_remote_and_endpoint_cases(monkeypatch, tmp_path):
    monkeypatch.setattr(hygiene, "ROOT", tmp_path)
    bad_lines = [
        "--" + "api" + "-key value",
        "bare " + "api" + "-key value",
        "ssh " + "ro" + "ot@host",
        'USER_NAME="${REMOTE_USER:-' + "ro" + "ot" + '}"',
        'USER_NAME="' + "ro" + "ot" + '"',
        'HOST="' + "203." + "0.113." + "9" + '"',
    ]

    path = tmp_path / "script.sh"
    path.write_text("\n".join(bad_lines) + "\n", encoding="utf-8")

    violations = hygiene.text_violations(path)

    assert len(violations) == len(bad_lines)
