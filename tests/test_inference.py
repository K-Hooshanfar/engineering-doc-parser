# tests/test_pipeline.py
# Pytest test suite with high coverage for the CLI pipeline.
# Assumes the pipeline module filename is `pipeline.py` located next to this tests/ folder.

import os
import sys

import numpy as np
import pytest

# --- Helpers / fakes ---------------------------------------------------------


class FakeTensor:
    def __init__(self, arr):
        self._arr = np.array(arr)

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self._arr)


class FakeBoxes:
    def __init__(self, xyxy=None, conf=None, cls=None):
        self.xyxy = FakeTensor(xyxy) if xyxy is not None else None
        self.conf = FakeTensor(conf) if conf is not None else None
        self.cls = FakeTensor(cls) if cls is not None else None


class FakeResult:
    def __init__(self, boxes: FakeBoxes):
        self.boxes = boxes


class FakeYOLO:
    """Minimal stand-in for ultralytics.YOLO.
    Call returns list with a single FakeResult; configure via ctor.
    """

    def __init__(self, *args, xyxy=None, conf=None, cls=None):
        # default one detection
        self.xyxy = np.array([[10, 10, 60, 50]]) if xyxy is None else np.array(xyxy)
        self.conf = np.array([0.9]) if conf is None else np.array(conf)
        self.cls = np.array([1]) if cls is None else np.array(cls)

    def __call__(self, image, **kwargs):
        return [FakeResult(FakeBoxes(self.xyxy, self.conf, self.cls))]


# Fake fitz (PyMuPDF) document tree for PDF tests
class _FakePixmap:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.samples = (
            np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
        ).tobytes()


class _FakePage:
    def __init__(self, w, h):
        self._w = w
        self._h = h

    def get_pixmap(self, matrix=None, alpha=False):
        return _FakePixmap(self._w, self._h)


class _FakeDoc:
    def __init__(self, n_pages=2, w=80, h=60):
        self.page_count = n_pages
        self._w = w
        self._h = h

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def load_page(self, i):
        return _FakePage(self._w, self._h)


# --- Fixtures ----------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_tmpdirs(tmp_path):
    # ensure any temp-created files are within pytest tmp
    cwd = os.getcwd()
    os.chdir(tmp_path)
    yield
    os.chdir(cwd)


@pytest.fixture
def pipeline_module(monkeypatch):
    """Import pipeline.py and expose it as module. Monkeypatch heavy deps.
    This assumes ultralytics + fitz exist in the environment; we still override usage.
    """

    # Import the module fresh each time to avoid cross-test state
    if "pipeline" in sys.modules:
        del sys.modules["pipeline"]
    import pipeline as P

    # Monkeypatch the heavy objects to our fakes
    monkeypatch.setattr(P, "YOLO", FakeYOLO, raising=True)

    # PyMuPDF: override open() and Matrix
    class _FakeMatrix:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(
        P.fitz, "open", lambda path: _FakeDoc(n_pages=2, w=64, h=48), raising=True
    )
    monkeypatch.setattr(P.fitz, "Matrix", _FakeMatrix, raising=True)

    return P


# --- Tests: naming & small utilities ----------------------------------------


def test_make_name_from_input_variants(pipeline_module):
    P = pipeline_module
    assert P.make_name_from_input("photo.jpg", keep_ext=False) == "photo"
    assert P.make_name_from_input("photo.jpg", keep_ext=True) == "photo.jpg"
    assert P.make_name_from_input("doc.pdf", keep_ext=False, page_index=0) == "doc_p000"
    assert (
        P.make_name_from_input("doc.pdf", keep_ext=True, page_index=12)
        == "doc.pdf_p012"
    )


def test_to_bgr8_conversions(pipeline_module):
    P = pipeline_module
    # grayscale uint8
    g = np.random.randint(0, 255, (10, 15), dtype=np.uint8)
    out = P._to_bgr8(g)
    assert out.shape == (10, 15, 3)
    assert out.dtype == np.uint8

    # float [0,1]
    f = np.random.rand(8, 9).astype(np.float32)
    out = P._to_bgr8(f)
    assert out.shape == (8, 9, 3)

    # uint16
    u16 = (np.random.rand(5, 7) * 50000).astype(np.uint16)
    out = P._to_bgr8(u16)
    assert out.dtype == np.uint8

    # RGBA -> BGR
    rgba = np.zeros((6, 6, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    out = P._to_bgr8(rgba)
    assert out.shape == (6, 6, 3)


def test_save_yolo_labels_content(tmp_path, pipeline_module):
    P = pipeline_module
    txt = tmp_path / "labels.txt"
    boxes = np.array([[10, 10, 60, 50], [0, 0, 100, 100]], dtype=float)
    clss = np.array([1, 2])
    P.save_yolo_labels(str(txt), boxes, clss, img_w=200, img_h=100, digits=6)
    s = txt.read_text().strip().splitlines()
    assert len(s) == 2
    # check normalized first line roughly
    cls, xc, yc, w, h = s[0].split()
    assert cls == "1"
    assert abs(float(xc) - 0.175) < 1e-4
    assert abs(float(yc) - 0.300) < 1e-4
    assert abs(float(w) - 0.25) < 1e-4
    assert abs(float(h) - 0.40) < 1e-4


def test_export_png_success_and_failure(tmp_path, monkeypatch, pipeline_module):
    P = pipeline_module
    arr = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
    outdir = tmp_path

    # success path using real cv2.imwrite
    p = P.export_png(arr, "img_base", str(outdir), 3)
    assert os.path.exists(p)

    # failure path by forcing cv2.imwrite to return False
    called = {"n": 0}

    def _fake_imwrite(path, data, params):
        called["n"] += 1
        return False

    monkeypatch.setattr(P.cv2, "imwrite", _fake_imwrite, raising=True)
    with pytest.raises(RuntimeError):
        P.export_png(arr, "img_fail", str(outdir), 3)
    assert called["n"] == 1


# --- Tests: image loading ----------------------------------------------------


def test_load_image_any_png_and_bgra(tmp_path, pipeline_module):
    P = pipeline_module
    # Write a simple PNG via OpenCV
    rgb = (np.random.rand(10, 12, 3) * 255).astype(np.uint8)
    bgr = rgb[:, :, ::-1]
    path_png = tmp_path / "x.png"
    assert P.cv2.imwrite(str(path_png), bgr)

    out = P.load_image_any(str(path_png))
    assert out.shape == (10, 12, 3)
    assert out.dtype == np.uint8

    # BGRA case
    bgra = np.dstack([bgr, np.full((10, 12), 255, dtype=np.uint8)])
    path_png4 = tmp_path / "x4.png"
    assert P.cv2.imwrite(str(path_png4), bgra)
    out4 = P.load_image_any(str(path_png4))
    assert out4.shape == (10, 12, 3)


def test_load_image_any_tiff_mock(monkeypatch, pipeline_module):
    P = pipeline_module

    class _FakePILImage:
        def __init__(self):
            self.n_frames = 3
            self.mode = "L"
            self._page = 0

        def seek(self, idx):
            self._page = idx

        def convert(self, mode):
            self.mode = mode
            return self

        def __array__(self, *args, **kwargs):
            # return a simple grayscale array; PIL->np uses __array__ via np.array(im)
            return np.random.randint(0, 255, (5, 7), dtype=np.uint8)

    def fake_open(path):
        return _FakePILImage()

    monkeypatch.setattr(P.Image, "open", staticmethod(fake_open), raising=True)

    out = P.load_image_any("dummy.tiff", page_index=1)
    assert out.shape[2] == 3
    assert out.dtype == np.uint8


# --- Tests: process_image_like ----------------------------------------------


def test_process_image_like_writes_labels_and_returns_crop(tmp_path, pipeline_module):
    P = pipeline_module

    img = np.zeros((100, 200, 3), dtype=np.uint8)
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    ok, cropped = P.process_image_like(
        img,
        "namebase",
        model=FakeYOLO(),
        labels_dir=str(labels_dir),
        conf_thresh=0.25,
        digits=6,
        write_empty_label=True,
    )
    assert ok is True
    assert isinstance(cropped, np.ndarray)
    # label file must exist
    txt = labels_dir / "namebase.txt"
    assert txt.exists()
    # crop dimensions reflect bbox (10,10)-(60,50)
    assert cropped.shape[0] == 40 and cropped.shape[1] == 50


def test_process_image_like_empty_and_below_threshold(tmp_path, pipeline_module):
    P = pipeline_module
    img = np.zeros((30, 40, 3), dtype=np.uint8)
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    # No detections (mask empty): conf very low
    ok, msg = P.process_image_like(
        img,
        "nolab",
        model=FakeYOLO(conf=[0.01]),
        labels_dir=str(labels_dir),
        conf_thresh=0.5,
        digits=6,
        write_empty_label=True,
    )
    assert ok is False
    assert (labels_dir / "nolab.txt").exists()
    assert (labels_dir / "nolab.txt").read_text().strip() == ""


# --- Tests: gather paths -----------------------------------------------------


def test_gather_paths_recursive_and_non_recursive(tmp_path, pipeline_module):
    P = pipeline_module
    (tmp_path / "a.png").write_bytes(b"0")
    (tmp_path / "b.jpg").write_bytes(b"0")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.png").write_bytes(b"0")

    nonrec = P.gather_paths(str(tmp_path), ["*.png"], recursive=False)
    assert len(nonrec) == 1

    rec = P.gather_paths(str(tmp_path), ["*.png"], recursive=True)
    # should include a.png and sub/c.png
    assert len(rec) == 2


# --- Tests: main() end-to-end (image path) ----------------------------------


def test_main_end_to_end_with_image(tmp_path, monkeypatch, pipeline_module):
    P = pipeline_module

    # Build a temp input dir with one image
    inp = tmp_path / "in"
    inp.mkdir()
    out_sub = "out"
    png_sub = "pngs"
    labels_sub = "labels"

    # create an image
    img = (np.random.rand(100, 200, 3) * 255).astype(np.uint8)
    P.cv2.imwrite(str(inp / "img.jpg"), img[:, :, ::-1])

    # FakeYOLO is already injected; build CLI args
    argv = [
        "pipeline.py",
        "--input-dir",
        str(inp),
        "--output-sub",
        out_sub,
        "--model-path",
        "dummy.pt",
        "--png-export-sub",
        png_sub,
        "--labels-sub",
        labels_sub,
        "--keep-input-extension-in-names",
        "0",
        "--recursive",
        "0",
    ]
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(sys, "argv", argv)

    P.main()

    # Assert outputs exist
    out_dir = inp / out_sub
    png_dir = inp / png_sub
    labels_dir = inp / labels_sub

    assert any(p.name.endswith("_crop.png") for p in out_dir.iterdir())
    assert any(p.name.endswith(".png") for p in png_dir.iterdir())
    assert any(p.name.endswith(".txt") for p in labels_dir.iterdir())


# --- Tests: main() PDF branch with faked fitz -------------------------------


def test_main_pdf_branch(tmp_path, monkeypatch, pipeline_module):
    P = pipeline_module

    # Prepare input dir with a dummy PDF file (content irrelevant—fitz is faked)
    inp = tmp_path / "in"
    inp.mkdir()
    pdf_path = inp / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%faked for tests\n")

    argv = [
        "pipeline.py",
        "--input-dir",
        str(inp),
        "--output-sub",
        "out",
        "--png-export-sub",
        "png",
        "--labels-sub",
        "labels",
        "--model-path",
        "dummy.pt",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    P.main()

    # Expect two pages processed by our fake doc
    out_dir = inp / "out"
    png_dir = inp / "png"
    labels_dir = inp / "labels"

    # PNG exports should include _p000.png and _p001.png
    png_names = {p.name for p in png_dir.iterdir()}
    assert any(n.endswith("_p000.png") for n in png_names)
    assert any(n.endswith("_p001.png") for n in png_names)

    # Crops and labels for both pages
    crop_names = {p.name for p in out_dir.iterdir()}
    assert any(n.endswith("_p000_crop.png") for n in crop_names)
    assert any(n.endswith("_p001_crop.png") for n in crop_names)

    lbl_names = {p.name for p in labels_dir.iterdir()}
    assert "doc_p000.txt" in lbl_names
    assert "doc_p001.txt" in lbl_names


# --- Tests: parse_args behaviour --------------------------------------------


def test_parse_args_sets_defaults_and_flags(monkeypatch, pipeline_module):
    P = pipeline_module

    argv = ["pipeline.py"]  # no args -> all defaults
    monkeypatch.setattr(sys, "argv", argv)
    ns = P.parse_args()
    assert ns.conf_thresh == 0.25
    assert ns.pdf_dpi == 200
    assert ns.keep_input_extension_in_names == 0

    argv = [
        "pipeline.py",
        "--conf-thresh",
        "0.5",
        "--pdf-dpi",
        "150",
        "--keep-input-extension-in-names",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    ns = P.parse_args()
    assert ns.conf_thresh == 0.5
    assert ns.pdf_dpi == 150
    assert ns.keep_input_extension_in_names == 1
