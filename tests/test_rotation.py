"""Tests for the `rotation` package.

This suite exercises orientation estimation, OSD parsing, rotation decisions,
directory iteration, and CLI flows. It stubs external deps (config, pytesseract)
to keep tests hermetic.
"""

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# -------------------------
# Test bootstrap / fixtures
# -------------------------


@pytest.fixture(scope="session")
def project_root():
    """Return the src directory added to sys.path for package imports."""
    return Path(__file__).resolve().parent.parent / "src"


@pytest.fixture(autouse=True, scope="session")
def add_project_to_path(project_root):
    """Prepend src/ to sys.path for the duration of the session."""
    sys.path.insert(0, str(project_root))
    yield


@pytest.fixture
def rotation_mod(monkeypatch):
    """Import orientation package with safe stubs for external dependencies.

    Stubs:
      - `config.Config` to satisfy `.config` import.
      - `pytesseract.image_to_osd` if pytesseract isn't present.
    """
    # Provide a dummy config module if missing
    cfg_mod = types.ModuleType("config")

    class _Config:
        EAST_TEXT_DETECTOR = "dummy_east.pb"

    cfg_mod.Config = _Config
    monkeypatch.setitem(sys.modules, "config", cfg_mod)

    # Provide a dummy pytesseract if it's not installed
    if "pytesseract" not in sys.modules:
        pt = types.ModuleType("pytesseract")

        def _image_to_osd(_img):
            return "Rotate: 0"

        pt.image_to_osd = _image_to_osd
        monkeypatch.setitem(sys.modules, "pytesseract", pt)

    # Import (or reload) the target module
    import engineering_doc_parser.orientation as rotation

    rotation = importlib.reload(rotation)
    return rotation


# --------------
# Unit tests
# --------------


def test_estimate_orientation_counts(rotation_mod):
    """Counts horizontal vs. vertical boxes; near-squares are ignored."""
    boxes = [
        (0, 0, 120, 10),  # horizontal (w/h = 12)
        (0, 0, 10, 120),  # vertical   (h/w = 12)
        (
            5,
            5,
            17,
            15,
        ),  # near-square: w=12, h=10 -> w/h = 1.2 (ignored; code uses > threshold)
    ]
    horiz, vert = rotation_mod.estimate_orientation(boxes, ratio_thresh=1.2)
    assert horiz == 1
    assert vert == 1


def test_detect_osd_rotation_parsing(rotation_mod, monkeypatch):
    """Parses rotation angle from Tesseract OSD output."""

    # Fake OSD output
    def fake_osd(_img):
        return "Orientation in degrees: 0\nRotate: 270\nSome other lines..."

    monkeypatch.setattr(rotation_mod.pytesseract, "image_to_osd", fake_osd)

    # The image isn't used by our fake, just pass a dummy array
    angle = rotation_mod.detect_osd_rotation(np.zeros((10, 10, 3), dtype=np.uint8))
    assert angle == 270


def test_rotate_if_needed_osd_90(rotation_mod, monkeypatch):
    """Rotates 90° CW when OSD suggests 90."""
    # Force OSD to request 90°
    monkeypatch.setattr(
        rotation_mod.pytesseract, "image_to_osd", lambda _img: "Rotate: 90"
    )

    img = np.zeros((10, 20, 3), dtype=np.uint8)  # H=10, W=20
    out, changed, msg = rotation_mod.rotate_if_needed(img, boxes=[], use_osd=True)
    # After 90° CW, shape should be (20, 10, 3)
    assert out.shape[:2] == (20, 10)
    assert changed is True
    assert "90°" in msg


def test_rotate_if_needed_east_fallback(rotation_mod, monkeypatch):
    """Falls back to EAST heuristic when OSD returns 0."""
    # OSD says 0 — fallback to EAST heuristic: more vertical boxes -> rotate CW
    monkeypatch.setattr(
        rotation_mod.pytesseract, "image_to_osd", lambda _img: "Rotate: 0"
    )

    img = np.zeros((20, 10, 3), dtype=np.uint8)
    boxes = [(0, 0, 10, 80)]  # tall box (vertical)
    out, changed, msg = rotation_mod.rotate_if_needed(
        img, boxes, use_osd=True, ratio_thresh=1.2
    )
    assert changed is True
    assert "EAST fallback" in msg
    # 90 CW: (H, W) -> (W, H)
    assert out.shape[:2] == (10, 20)


def test_iter_image_files_nonrecursive_and_recursive(tmp_path, rotation_mod):
    """Finds image files with and without recursion, filters non-images."""
    # Create files
    (tmp_path / "a.png").write_bytes(b"")
    (tmp_path / "b.jpg").write_bytes(b"")
    (tmp_path / "c.txt").write_text("ignore me")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.png").write_bytes(b"")

    # Non-recursive
    files_nonrec = list(
        rotation_mod.iter_image_files(str(tmp_path), ["png", "jpg"], recursive=False)
    )
    assert set(map(Path, files_nonrec)) == {tmp_path / "a.png", tmp_path / "b.jpg"}

    # Recursive
    files_rec = list(
        rotation_mod.iter_image_files(str(tmp_path), ["png", "jpg"], recursive=True)
    )
    assert set(map(Path, files_rec)) == {
        tmp_path / "a.png",
        tmp_path / "b.jpg",
        sub / "c.png",
    }


def test_process_one_image_rotated_and_saved(tmp_path, rotation_mod, monkeypatch):
    """Saves a new file with `_rotated` suffix when a rotation occurs."""
    # Fake image read
    monkeypatch.setattr(
        rotation_mod.cv2, "imread", lambda p: np.zeros((10, 10, 3), dtype=np.uint8)
    )

    # Avoid running EAST/Tesseract: stub detect + rotate
    monkeypatch.setattr(
        rotation_mod, "detect_east_boxes", lambda img, net, conf_thresh, nms_thresh: []
    )

    def fake_rotate(img, boxes, **kwargs):
        return img.copy(), True, "Rotated 90° CCW"

    monkeypatch.setattr(rotation_mod, "rotate_if_needed", fake_rotate)

    # Capture saves
    saved = []

    def fake_save(img, output_path):
        saved.append(output_path)

    monkeypatch.setattr(rotation_mod, "save_image", fake_save)

    # Run
    dummy_net = object()
    in_path = tmp_path / "scan1.png"
    in_path.write_bytes(b"fake")
    rotation_mod.process_one_image(
        image_path=str(in_path),
        net=dummy_net,
        result_dir=str(tmp_path / "out"),
        conf_thresh=0.5,
        nms_thresh=0.4,
        ratio_thresh=1.2,
        use_osd=True,
        force_angle=None,
        save_unchanged=False,
        verbose=False,
    )

    assert len(saved) == 1
    assert saved[0].endswith("scan1_rotated.png")


def test_process_one_image_unchanged_saved_when_flag(
    tmp_path, rotation_mod, monkeypatch
):
    """Saves original filename when no rotation and `save_unchanged=True`."""
    # Fake image read
    monkeypatch.setattr(
        rotation_mod.cv2, "imread", lambda p: np.zeros((10, 10, 3), dtype=np.uint8)
    )

    # No rotation needed
    monkeypatch.setattr(
        rotation_mod, "detect_east_boxes", lambda img, net, conf_thresh, nms_thresh: []
    )

    def fake_rotate(img, boxes, **kwargs):
        return img, False, "No orientation needed"

    monkeypatch.setattr(rotation_mod, "rotate_if_needed", fake_rotate)

    saved = []
    monkeypatch.setattr(rotation_mod, "save_image", lambda img, out: saved.append(out))

    dummy_net = object()
    in_path = tmp_path / "doc.jpg"
    in_path.write_bytes(b"fake")
    rotation_mod.process_one_image(
        image_path=str(in_path),
        net=dummy_net,
        result_dir=str(tmp_path / "res"),
        conf_thresh=0.5,
        nms_thresh=0.4,
        ratio_thresh=1.2,
        use_osd=True,
        force_angle=None,
        save_unchanged=True,
        verbose=False,
    )

    assert len(saved) == 1
    assert saved[0].endswith("doc.jpg")  # unchanged save keeps original name


def test_main_single_image_flow(rotation_mod, monkeypatch, tmp_path):
    """Drives `main()` in single-image mode and verifies argument plumbing."""
    # Stub argparse returns
    args = SimpleNamespace(
        image_path=str(tmp_path / "one.png"),
        input_dir=None,
        east_model=str(tmp_path / "east.pb"),
        result_dir=str(tmp_path / "out"),
        conf_thresh=0.5,
        nms_thresh=0.4,
        ratio_thresh=1.2,
        no_osd=False,
        force_angle=None,
        recursive=False,
        exts=["png", "jpg"],
        save_unchanged=False,
        verbose=False,
        log_level=None,
        log_file=None,
    )
    # Write dummy files that main() checks
    Path(args.image_path).write_bytes(b"img")
    Path(args.east_model).write_bytes(b"pb")

    monkeypatch.setattr(
        rotation_mod.argparse.ArgumentParser, "parse_args", lambda self=None: args
    )

    # Pretend these exist
    monkeypatch.setattr(rotation_mod.os.path, "isfile", lambda p: True)

    # Stub readNet
    monkeypatch.setattr(rotation_mod.cv2.dnn, "readNet", lambda p: object())

    # Capture process_one_image invocation
    called = {}

    def fake_process(img_path, net, **kwargs):
        called["img_path"] = img_path
        called["kwargs"] = kwargs

    monkeypatch.setattr(rotation_mod, "process_one_image", fake_process)

    rotation_mod.main()
    assert called["img_path"] == args.image_path
    assert called["kwargs"]["result_dir"] == args.result_dir


def test_main_directory_flow(rotation_mod, monkeypatch, tmp_path):
    """Drives `main()` in directory mode and ensures each image path is processed."""
    # Prepare args for directory mode
    args = SimpleNamespace(
        image_path=str(tmp_path / "unused.png"),
        input_dir=str(tmp_path / "scans"),
        east_model=str(tmp_path / "east.pb"),
        result_dir=str(tmp_path / "out"),
        conf_thresh=0.5,
        nms_thresh=0.4,
        ratio_thresh=1.2,
        no_osd=False,
        force_angle=None,
        recursive=True,
        exts=["png", "jpg"],
        save_unchanged=False,
        verbose=False,
        log_level=None,
        log_file=None,
    )
    Path(args.input_dir).mkdir(parents=True, exist_ok=True)
    Path(args.east_model).write_bytes(b"pb")

    monkeypatch.setattr(
        rotation_mod.argparse.ArgumentParser, "parse_args", lambda self=None: args
    )
    monkeypatch.setattr(rotation_mod.os.path, "isfile", lambda p: True)
    monkeypatch.setattr(rotation_mod.os.path, "isdir", lambda p: True)
    monkeypatch.setattr(rotation_mod.cv2.dnn, "readNet", lambda p: object())

    # Make iter_image_files return some files
    files = [str(Path(args.input_dir) / "a.png"), str(Path(args.input_dir) / "b.jpg")]
    monkeypatch.setattr(
        rotation_mod, "iter_image_files", lambda root, exts, recursive: files
    )

    called = []

    def fake_process(img_path, net, **kwargs):
        called.append(img_path)

    monkeypatch.setattr(rotation_mod, "process_one_image", fake_process)

    rotation_mod.main()
    assert called == files
