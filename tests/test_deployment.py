"""Check saved model and deployment configuration consistency."""

import json
import re
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_model_info():
    with (ROOT / "models" / "model_info.json").open(encoding="utf-8") as file:
        return json.load(file)


def read_runtime_requirements():
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    return [
        line.split("#", 1)[0].strip()
        for line in text.splitlines()
        if line.split("#", 1)[0].strip()
    ]


def read_space_header():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    lines = text.splitlines()

    assert lines and lines[0].strip() == "---", (
        "README.md must start with the Hugging Face YAML header."
    )

    closing = next(
        (index for index in range(1, len(lines)) if lines[index].strip() == "---"),
        None,
    )
    assert closing is not None, "The YAML header needs a closing ---."

    header = {}
    for line in lines[1:closing]:
        if ":" in line and not line.lstrip().startswith("#"):
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip().strip("\"'")

    return header


def test_model_artifacts_exist():
    assert (ROOT / "models" / "late_orders.joblib").is_file()
    assert (ROOT / "models" / "model_info.json").is_file()


def test_installed_sklearn_matches_saved_model():
    info = read_model_info()

    assert version("scikit-learn") == info["scikit_learn_version"]


def test_requirements_pin_matches_saved_sklearn():
    info = read_model_info()
    requirements = read_runtime_requirements()

    assert f"scikit-learn=={info['scikit_learn_version']}" in requirements


def test_runtime_dependencies_are_pinned():
    requirements = read_runtime_requirements()

    for name in ("scikit-learn", "pandas", "joblib"):
        matches = [
            line
            for line in requirements
            if re.fullmatch(
                rf"{re.escape(name)}==[^\s;]+",
                line,
                flags=re.IGNORECASE,
            )
        ]
        assert len(matches) == 1, f"{name} needs one exact version pin."


def test_gradio_is_not_in_runtime_requirements():
    requirements = read_runtime_requirements()

    assert not any(
        re.match(r"gradio(?:$|[\s<>=!~;\[])", line, re.IGNORECASE)
        for line in requirements
    )


def test_space_uses_gradio():
    header = read_space_header()

    assert header.get("sdk") == "gradio"


def test_space_gradio_version_matches_installed_version():
    header = read_space_header()

    assert header.get("sdk_version") == version("gradio")


def test_space_app_file_exists():
    header = read_space_header()
    app_file = header.get("app_file")

    assert app_file, "The YAML header must specify app_file."
    assert (ROOT / app_file).is_file()
