"""
Tests for the `engineering_doc_parser.table_cropper` module.

This suite tests YOLO-based rotation detection, image processing,
scoring functions, and batch processing. It mocks external dependencies
(YOLO models, file I/O) to keep tests fast and hermetic.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import engineering_doc_parser
from engineering_doc_parser.table_cropper import cropper


@pytest.fixture
def sample_image_bgr():
    """Create a sample BGR image (100x200x3)."""
    return np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_bytes(sample_image_bgr):
    """Convert sample image to PNG bytes."""
    import cv2

    _, buf = cv2.imencode(".png", sample_image_bgr)
    return buf.tobytes()


@pytest.fixture
def mock_yolo_model():
    """Create a mock YOLO model that returns detection results."""
    model = MagicMock()

    # Mock result object
    result = MagicMock()
    result.boxes = MagicMock()
    result.boxes.xyxy = np.array(
        [[10, 20, 50, 80], [60, 30, 90, 100]], dtype=np.float32
    )
    result.boxes.conf = np.array([0.85, 0.75], dtype=np.float32)
    result.boxes.cls = np.array([0, 1], dtype=np.int32)

    # Make tensors have .cpu() and .numpy() methods
    result.boxes.xyxy = MagicMock()
    result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array(
        [[10, 20, 50, 80], [60, 30, 90, 100]], dtype=np.float32
    )
    result.boxes.conf.cpu.return_value.numpy.return_value = np.array(
        [0.85, 0.75], dtype=np.float32
    )
    result.boxes.cls.cpu.return_value.numpy.return_value = np.array(
        [0, 1], dtype=np.int32
    )

    # Simpler: just make them numpy arrays directly
    result.boxes.xyxy = np.array(
        [[10, 20, 50, 80], [60, 30, 90, 100]], dtype=np.float32
    )
    result.boxes.conf = np.array([0.85, 0.75], dtype=np.float32)
    result.boxes.cls = np.array([0, 1], dtype=np.int32)

    model.return_value = [result]
    return model


class TestImageIO:
    """Tests for image loading and encoding functions."""

    def test_to_bgr8_uint8_grayscale(self):
        """Test conversion of grayscale uint8 to BGR."""
        _to_bgr8 = engineering_doc_parser.table_cropper.cropper._to_bgr8

        gray = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        bgr = _to_bgr8(gray)

        assert bgr.shape == (50, 50, 3)
        assert bgr.dtype == np.uint8
        assert np.array_equal(bgr[:, :, 0], bgr[:, :, 1])  # B=G=R for grayscale
        assert np.array_equal(bgr[:, :, 1], bgr[:, :, 2])

    def test_to_bgr8_float_normalized(self):
        """Test conversion of float [0,1] to BGR."""
        _to_bgr8 = engineering_doc_parser.table_cropper.cropper._to_bgr8

        float_img = np.random.rand(50, 50, 3).astype(np.float32)
        bgr = _to_bgr8(float_img)

        assert bgr.shape == (50, 50, 3)
        assert bgr.dtype == np.uint8
        assert np.all(bgr >= 0) and np.all(bgr <= 255)

    def test_to_bgr8_uint16(self):
        """Test conversion of uint16 to BGR."""
        _to_bgr8 = engineering_doc_parser.table_cropper.cropper._to_bgr8

        uint16_img = np.random.randint(0, 65535, (50, 50, 3), dtype=np.uint16)
        bgr = _to_bgr8(uint16_img)

        assert bgr.shape == (50, 50, 3)
        assert bgr.dtype == np.uint8

    def test_load_image_from_bytes_png(self, sample_image_bytes):
        """Test loading image from PNG bytes."""
        load_image_from_bytes = engineering_doc_parser.table_cropper.cropper.load_image_from_bytes

        img = load_image_from_bytes(sample_image_bytes)

        assert img.ndim == 3
        assert img.shape[2] == 3
        assert img.dtype == np.uint8

    def test_encode_png_bytes(self, sample_image_bgr):
        """Test encoding BGR image to PNG bytes."""
        encode_png_bytes = engineering_doc_parser.table_cropper.cropper.encode_png_bytes

        png_bytes = encode_png_bytes(sample_image_bgr)

        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"  # PNG signature


class TestRotation:
    """Tests for rotation functions."""

    def test_rotate_image_0_degrees(self, sample_image_bgr):
        """Test rotation by 0 degrees (no change)."""
        rotate_image = engineering_doc_parser.table_cropper.cropper.rotate_image

        rotated = rotate_image(sample_image_bgr, 0)

        assert np.array_equal(rotated, sample_image_bgr)
        assert rotated is not sample_image_bgr  # Should be a copy

    def test_rotate_image_90_degrees(self, sample_image_bgr):
        """Test rotation by 90 degrees."""
        rotate_image = engineering_doc_parser.table_cropper.cropper.rotate_image

        h, w = sample_image_bgr.shape[:2]
        rotated = rotate_image(sample_image_bgr, 90)

        assert rotated.shape == (w, h, 3)

    def test_rotate_image_180_degrees(self, sample_image_bgr):
        """Test rotation by 180 degrees."""
        rotate_image = engineering_doc_parser.table_cropper.cropper.rotate_image

        rotated = rotate_image(sample_image_bgr, 180)

        assert rotated.shape == sample_image_bgr.shape

    def test_rotate_image_270_degrees(self, sample_image_bgr):
        """Test rotation by 270 degrees."""
        rotate_image = engineering_doc_parser.table_cropper.cropper.rotate_image

        h, w = sample_image_bgr.shape[:2]
        rotated = rotate_image(sample_image_bgr, 270)

        assert rotated.shape == (w, h, 3)


class TestScoring:
    """Tests for scoring and utility functions."""

    def test_boxes_area_xyxy(self):
        """Test bounding box area calculation."""
        _boxes_area_xyxy = engineering_doc_parser.table_cropper.cropper._boxes_area_xyxy

        boxes = np.array([[0, 0, 10, 20], [5, 5, 15, 25]], dtype=np.float32)
        areas = _boxes_area_xyxy(boxes)

        assert len(areas) == 2
        assert areas[0] == 200.0  # 10 * 20
        assert areas[1] == 200.0  # 10 * 20

    def test_union_coverage(self):
        """Test union coverage calculation."""
        _union_coverage = engineering_doc_parser.table_cropper.cropper._union_coverage

        boxes = np.array([[10, 10, 50, 50], [30, 30, 70, 70]], dtype=np.float32)
        coverage = _union_coverage(boxes, img_w=100, img_h=100)

        assert 0.0 <= coverage <= 1.0
        assert coverage > 0.0  # Should have some coverage

    def test_fragmentation_penalty(self):
        """Test fragmentation penalty calculation."""
        _fragmentation_penalty = engineering_doc_parser.table_cropper.cropper._fragmentation_penalty

        # Many small boxes (high fragmentation)
        small_areas = np.array([10, 15, 12, 8, 20], dtype=np.float32)
        frag_high = _fragmentation_penalty(small_areas)

        # Few large boxes (low fragmentation)
        large_areas = np.array([1000, 2000], dtype=np.float32)
        frag_low = _fragmentation_penalty(large_areas)

        assert 0.0 <= frag_high <= 1.0
        assert 0.0 <= frag_low <= 1.0
        assert frag_high > frag_low  # Small boxes should have higher penalty

    def test_compute_bottom_bias_score(self):
        """Test bottom bias score calculation."""
        compute_bottom_bias_score = engineering_doc_parser.table_cropper.cropper.compute_bottom_bias_score

        # Boxes near bottom
        boxes_bottom = np.array([[10, 80, 50, 95], [60, 85, 90, 98]], dtype=np.float32)
        score_bottom = compute_bottom_bias_score(boxes_bottom, img_height=100)

        # Boxes near top
        boxes_top = np.array([[10, 5, 50, 20], [60, 10, 90, 25]], dtype=np.float32)
        score_top = compute_bottom_bias_score(boxes_top, img_height=100)

        assert 0.0 <= score_bottom <= 1.0
        assert 0.0 <= score_top <= 1.0
        assert score_bottom > score_top  # Bottom boxes should score higher


class TestDetection:
    """Tests for YOLO detection functions."""

    @patch("engineering_doc_parser.table_cropper.cropper._get_model")
    def test_detect_with_model_success(
        self, mock_get_model, sample_image_bgr, mock_yolo_model
    ):
        """Test successful detection with mock model."""
        detect_with_model = engineering_doc_parser.table_cropper.cropper.detect_with_model

        mock_get_model.return_value = mock_yolo_model
        mock_yolo_model.return_value = [
            MagicMock(
                boxes=MagicMock(
                    xyxy=np.array([[10, 20, 50, 80]], dtype=np.float32),
                    conf=np.array([0.85], dtype=np.float32),
                    cls=np.array([0], dtype=np.int32),
                )
            )
        ]

        # Make boxes accessible as numpy arrays
        result = mock_yolo_model.return_value[0]
        result.boxes.xyxy = np.array([[10, 20, 50, 80]], dtype=np.float32)
        result.boxes.conf = np.array([0.85], dtype=np.float32)
        result.boxes.cls = np.array([0], dtype=np.int32)

        success, cropped, avg_conf, num_boxes, boxes, confs, clss = detect_with_model(
            model_path="dummy.onnx", image=sample_image_bgr, conf_thresh=0.5
        )

        assert success is True
        assert num_boxes > 0
        assert avg_conf > 0.0
        assert boxes.shape[0] > 0

    @patch("engineering_doc_parser.table_cropper.cropper._get_model")
    def test_detect_with_model_no_detections(self, mock_get_model, sample_image_bgr):
        """Test detection when no boxes are found."""
        detect_with_model = engineering_doc_parser.table_cropper.cropper.detect_with_model

        mock_model = MagicMock()
        result = MagicMock()
        result.boxes = None
        mock_model.return_value = [result]
        mock_get_model.return_value = mock_model

        success, cropped, avg_conf, num_boxes, boxes, confs, clss = detect_with_model(
            model_path="dummy.onnx", image=sample_image_bgr, conf_thresh=0.5
        )

        assert success is False
        assert num_boxes == 0
        assert cropped is None


class TestFindBestRotation:
    """Tests for rotation finding logic."""

    @patch("engineering_doc_parser.table_cropper.cropper.detect_with_model")
    def test_find_best_rotation_all_angles(self, mock_detect, sample_image_bgr):
        """Test finding best rotation across all angles."""
        find_best_rotation = engineering_doc_parser.table_cropper.cropper.find_best_rotation

        # Mock detection results for different rotations
        def mock_detect_side_effect(model_path, image, conf_thresh):
            # Return different scores for different rotations
            # Simulate that 0° has best detection
            h, w = image.shape[:2]
            if image.shape == sample_image_bgr.shape:  # 0°
                boxes = np.array([[10, 20, 50, 80]], dtype=np.float32)
                confs = np.array([0.9], dtype=np.float32)
                clss = np.array([0], dtype=np.int32)
            else:  # Rotated
                boxes = np.array([[5, 10, 30, 40]], dtype=np.float32)
                confs = np.array([0.6], dtype=np.float32)
                clss = np.array([0], dtype=np.int32)

            cropped = image[20:80, 10:50] if boxes.shape[0] > 0 else None
            return (
                True,
                cropped,
                float(np.mean(confs)),
                boxes.shape[0],
                boxes,
                confs,
                clss,
            )

        mock_detect.side_effect = mock_detect_side_effect

        angle, rotated, cropped = find_best_rotation(
            image=sample_image_bgr,
            model_path="dummy.onnx",
            conf_thresh=0.5,
            verbose=False,
        )

        assert angle in [0, 90, 180, 270]
        assert rotated is not None
        assert (
            rotated.shape == sample_image_bgr.shape
            or rotated.shape[:2] == sample_image_bgr.shape[:2][::-1]
        )

    @patch("engineering_doc_parser.table_cropper.cropper.detect_with_model")
    def test_find_best_rotation_debug_output(self, mock_detect, sample_image_bgr):
        """Test rotation finding with debug output."""
        find_best_rotation = engineering_doc_parser.table_cropper.cropper.find_best_rotation

        def mock_detect_side_effect(model_path, image, conf_thresh):
            boxes = np.array([[10, 20, 50, 80]], dtype=np.float32)
            confs = np.array([0.85], dtype=np.float32)
            clss = np.array([0], dtype=np.int32)
            cropped = image[20:80, 10:50] if boxes.shape[0] > 0 else None
            return True, cropped, 0.85, 1, boxes, confs, clss

        mock_detect.side_effect = mock_detect_side_effect

        rotation_candidates_debug = []
        rotation_debug_images = []

        angle, rotated, cropped = find_best_rotation(
            image=sample_image_bgr,
            model_path="dummy.onnx",
            conf_thresh=0.5,
            verbose=False,
            rotation_candidates_debug=rotation_candidates_debug,
            rotation_debug_images=rotation_debug_images,
        )

        assert len(rotation_candidates_debug) == 4  # One for each angle
        assert len(rotation_debug_images) == 4
        assert all("angle" in d for d in rotation_candidates_debug)
        assert all("score" in d for d in rotation_candidates_debug)


class TestCropTables:
    """Tests for main cropping functions."""

    @patch("engineering_doc_parser.table_cropper.cropper.find_best_rotation")
    @patch("engineering_doc_parser.table_cropper.cropper.load_image_from_bytes")
    def test_crop_tables_from_bytes(
        self, mock_load, mock_find_rotation, sample_image_bytes
    ):
        """Test cropping tables from image bytes."""
        crop_tables_from_bytes = engineering_doc_parser.table_cropper.cropper.crop_tables_from_bytes

        mock_load.return_value = np.random.randint(
            0, 255, (100, 200, 3), dtype=np.uint8
        )

        # Mock successful rotation and cropping
        rotated_img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        cropped_img = np.random.randint(0, 255, (50, 100, 3), dtype=np.uint8)
        mock_find_rotation.return_value = (0, rotated_img, cropped_img)

        result = crop_tables_from_bytes(
            image_bytes=sample_image_bytes, conf_thresh=0.5, verbose=False
        )

        assert result is not None
        assert result.ndim == 3
        assert result.shape[2] == 3

    @patch("engineering_doc_parser.table_cropper.cropper.crop_tables_from_bytes")
    def test_crop_tables_from_bytes_png(self, mock_crop, sample_image_bgr):
        """Test cropping and encoding to PNG bytes."""
        crop_tables_from_bytes_png = engineering_doc_parser.table_cropper.cropper.crop_tables_from_bytes_png

        mock_crop.return_value = sample_image_bgr

        png_bytes = crop_tables_from_bytes_png(
            image_bytes=b"dummy", conf_thresh=0.5, verbose=False
        )

        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0


class TestBatchProcessing:
    """Tests for batch processing functions."""

    def test_collect_files(self, tmp_path):
        """Test file collection from directory."""
        collect_files = engineering_doc_parser.table_cropper.cropper.collect_files

        # Create test files
        (tmp_path / "image1.png").write_bytes(b"dummy")
        (tmp_path / "image2.jpg").write_bytes(b"dummy")
        (tmp_path / "document.txt").write_text("ignore")
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "image3.tif").write_bytes(b"dummy")

        # Non-recursive
        files = collect_files(tmp_path, include_subdirs=False)
        assert len(files) == 2
        assert all(
            f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
            for f in files
        )

        # Recursive
        files_rec = collect_files(tmp_path, include_subdirs=True)
        assert len(files_rec) == 3

    @patch("engineering_doc_parser.table_cropper.cropper.crop_tables_from_bytes_png")
    @patch("engineering_doc_parser.table_cropper.cropper.collect_files")
    def test_process_folder(self, mock_collect, mock_crop, tmp_path):
        """Test folder processing."""
        process_folder = engineering_doc_parser.table_cropper.cropper.process_folder

        # Setup mocks
        test_files = [tmp_path / "img1.png", tmp_path / "img2.jpg"]
        for f in test_files:
            f.write_bytes(b"dummy")
        mock_collect.return_value = test_files

        # Mock successful cropping
        import cv2

        dummy_img = np.zeros((50, 50, 3), dtype=np.uint8)
        _, png_bytes = cv2.imencode(".png", dummy_img)
        mock_crop.return_value = png_bytes.tobytes()

        output_dir = tmp_path / "output"
        process_folder(
            input_dir=str(tmp_path),
            output_dir=str(output_dir),
            conf_thresh=0.5,
            verbose=False,
            show_progress=False,
        )

        assert output_dir.exists()
        mock_collect.assert_called_once()
        assert mock_crop.call_count == len(test_files)


class TestCLI:
    """Tests for CLI argument parsing."""

    def test_parse_args_defaults(self):
        """Test argument parser with defaults."""
        parse_args = engineering_doc_parser.table_cropper.cropper.parse_args

        # Mock sys.argv
        with patch("sys.argv", ["cropper.py"]):
            args = parse_args()

            assert args.input_path == "input_images"
            assert args.output_dir == "output_crops"
            assert args.conf_thresh == 0.25
            assert args.prefer_bottom is True

    def test_parse_args_custom(self):
        """Test argument parser with custom values."""
        parse_args = engineering_doc_parser.table_cropper.cropper.parse_args

        with patch(
            "sys.argv",
            [
                "cropper.py",
                "--in",
                "custom_input",
                "--out",
                "custom_output",
                "--conf",
                "0.3",
                "--no-bottom-bias",
                "--rotation-debug",
            ],
        ):
            args = parse_args()

            assert args.input_path == "custom_input"
            assert args.output_dir == "custom_output"
            assert args.conf_thresh == 0.3
            assert args.prefer_bottom is False
            assert args.rotation_debug is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
