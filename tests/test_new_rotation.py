"""
Tests for the `engineering_doc_parser.rotation` module.

This suite tests EasyOCR-based rotation detection, OCR scoring,
image preprocessing, and batch processing. It mocks external dependencies
(EasyOCR, Tesseract) to keep tests fast and hermetic.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def sample_image_bgr():
    """Create a sample BGR image (100x200x3)."""
    return np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)


@pytest.fixture
def mock_easyocr_reader():
    """Create a mock EasyOCR Reader."""
    reader = MagicMock()

    # Mock readtext results - horizontal text (good orientation)
    good_result = [
        ([[10, 20], [50, 20], [50, 40], [10, 40]], "Hello", 0.9),
        ([[60, 30], [120, 30], [120, 50], [60, 50]], "World", 0.85),
    ]

    # Mock readtext results - vertical text (bad orientation)
    bad_result = [
        ([[10, 10], [20, 10], [20, 80], [10, 80]], "H", 0.5),
    ]

    def readtext_side_effect(img, **kwargs):
        # Simple heuristic: if image is wide (horizontal), return good result
        h, w = img.shape[:2]
        if w > h:  # Horizontal image
            return good_result
        else:  # Vertical image
            return bad_result

    reader.readtext.side_effect = readtext_side_effect
    return reader


@pytest.fixture
def mock_tesseract_osd():
    """Create a mock Tesseract OSD function."""

    def fake_osd(img):
        return "Rotate: 0\nOrientation confidence: 15.5"

    return fake_osd


class TestRotation:
    """Tests for rotation functions."""

    def test_rotate_image_0_degrees(self, sample_image_bgr):
        """Test rotation by 0 degrees (no change)."""
        import engineering_doc_parser.rotation as new_rotation

        rotate_image = new_rotation.rotate_image

        rotated = rotate_image(sample_image_bgr, 0)

        assert np.array_equal(rotated, sample_image_bgr)

    def test_rotate_image_90_degrees(self, sample_image_bgr):
        """Test rotation by 90 degrees."""
        import engineering_doc_parser.rotation as new_rotation

        rotate_image = new_rotation.rotate_image

        h, w = sample_image_bgr.shape[:2]
        rotated = rotate_image(sample_image_bgr, 90)

        assert rotated.shape == (w, h, 3)

    def test_rotate_image_180_degrees(self, sample_image_bgr):
        """Test rotation by 180 degrees."""
        from engineering_doc_parser.rotation.core import rotate_image

        rotated = rotate_image(sample_image_bgr, 180)

        assert rotated.shape == sample_image_bgr.shape

    def test_rotate_image_270_degrees(self, sample_image_bgr):
        """Test rotation by 270 degrees."""
        from engineering_doc_parser.rotation.core import rotate_image

        h, w = sample_image_bgr.shape[:2]
        rotated = rotate_image(sample_image_bgr, 270)

        assert rotated.shape == (w, h, 3)

    def test_rotate_image_invalid_angle(self, sample_image_bgr):
        """Test rotation with invalid angle raises error."""
        from engineering_doc_parser.rotation.core import rotate_image

        with pytest.raises(ValueError):
            rotate_image(sample_image_bgr, 45)


class TestOSDParsing:
    """Tests for Tesseract OSD parsing."""

    def test_parse_osd_standard_format(self):
        """Test parsing standard OSD output."""
        from engineering_doc_parser.rotation.core import parse_osd

        osd_text = "Rotate: 90\nOrientation confidence: 18.5"
        angle, conf = parse_osd(osd_text)

        assert angle == 90
        assert conf == 18.5

    def test_parse_osd_orientation_format(self):
        """Test parsing OSD output with 'Orientation in degrees' format."""
        from engineering_doc_parser.rotation.core import parse_osd

        osd_text = "Orientation in degrees: 180\nOrientation confidence: 20.0"
        angle, conf = parse_osd(osd_text)

        assert angle == 180
        assert conf == 20.0

    def test_parse_osd_no_match(self):
        """Test parsing OSD output with no matches."""
        from engineering_doc_parser.rotation.core import parse_osd

        osd_text = "No rotation information"
        angle, conf = parse_osd(osd_text)

        assert angle is None
        assert conf is None

    @patch("engineering_doc_parser.rotation.core.pytesseract")
    def test_tesseract_osd_angle_conf_success(self, mock_pytesseract, sample_image_bgr):
        """Test successful Tesseract OSD call."""
        from engineering_doc_parser.rotation.core import tesseract_osd_angle_conf

        mock_pytesseract.image_to_osd.return_value = (
            "Rotate: 270\nOrientation confidence: 15.0"
        )

        angle, conf = tesseract_osd_angle_conf(sample_image_bgr)

        assert angle == 270
        assert conf == 15.0

    @patch("engineering_doc_parser.rotation.core.pytesseract")
    def test_tesseract_osd_angle_conf_exception(
        self, mock_pytesseract, sample_image_bgr
    ):
        """Test Tesseract OSD call with exception."""
        from engineering_doc_parser.rotation.core import tesseract_osd_angle_conf

        mock_pytesseract.image_to_osd.side_effect = Exception("Tesseract error")

        angle, conf = tesseract_osd_angle_conf(sample_image_bgr)

        assert angle is None
        assert conf is None


class TestBBoxDimensions:
    """Tests for bounding box dimension calculations."""

    def test_get_bbox_dimensions_horizontal(self):
        """Test dimension calculation for horizontal box."""
        from engineering_doc_parser.rotation.core import get_bbox_dimensions

        bbox = [[0, 0], [100, 0], [100, 20], [0, 20]]
        width, height = get_bbox_dimensions(bbox)

        assert width > height
        assert abs(width - 100.0) < 1.0
        assert abs(height - 20.0) < 1.0

    def test_get_bbox_dimensions_vertical(self):
        """Test dimension calculation for vertical box."""
        from engineering_doc_parser.rotation.core import get_bbox_dimensions

        bbox = [[0, 0], [20, 0], [20, 100], [0, 100]]
        width, height = get_bbox_dimensions(bbox)

        assert height > width
        assert abs(width - 20.0) < 1.0
        assert abs(height - 100.0) < 1.0


class TestScoring:
    """Tests for OCR scoring functions."""

    def test_calculate_orientation_score_good_text(self):
        """Test scoring with high-confidence horizontal text."""
        from engineering_doc_parser.rotation.core import calculate_orientation_score

        # High-confidence horizontal text
        result = [
            ([[0, 0], [100, 0], [100, 20], [0, 20]], "Hello World", 0.9),
            ([[0, 30], [80, 30], [80, 50], [0, 50]], "Test Text", 0.85),
        ]

        score = calculate_orientation_score(result, conf_th=0.45)

        assert score > 0  # Should be positive for good text

    def test_calculate_orientation_score_vertical_text(self):
        """Test scoring with vertical text (should score lower)."""
        from engineering_doc_parser.rotation.core import calculate_orientation_score

        # Vertical text (bad orientation)
        result = [
            ([[0, 0], [20, 0], [20, 100], [0, 100]], "H", 0.6),
        ]

        score = calculate_orientation_score(result, conf_th=0.45)

        # Vertical text should score lower than horizontal
        assert score < 1000  # Lower than typical good horizontal text

    def test_calculate_orientation_score_empty(self):
        """Test scoring with empty result."""
        from engineering_doc_parser.rotation.core import calculate_orientation_score

        score = calculate_orientation_score([], conf_th=0.45)

        assert score == -1e9  # Should return very negative score

    def test_calculate_orientation_score_low_confidence(self):
        """Test scoring with low-confidence text."""
        from engineering_doc_parser.rotation.core import calculate_orientation_score

        # Low confidence text
        result = [
            ([[0, 0], [50, 0], [50, 15], [0, 15]], "abc", 0.3),
        ]

        score = calculate_orientation_score(result, conf_th=0.45)

        # May include low-conf items if very few high-conf, but score should be lower
        assert isinstance(score, float)

    def test_calculate_orientation_score_word_patterns(self):
        """Test scoring with word-like patterns."""
        from engineering_doc_parser.rotation.core import calculate_orientation_score

        # Alphanumeric words
        result = [
            ([[0, 0], [100, 0], [100, 20], [0, 20]], "ABC123", 0.8),
            ([[0, 30], [80, 30], [80, 50], [0, 50]], "TEST", 0.85),  # All caps
        ]

        score = calculate_orientation_score(result, conf_th=0.45)

        assert score > 0  # Should get bonus for word patterns


class TestPredictRotation:
    """Tests for rotation prediction."""

    @patch("engineering_doc_parser.rotation.core.easyocr.Reader")
    @patch("engineering_doc_parser.rotation.core.tesseract_osd_angle_conf")
    @patch("engineering_doc_parser.rotation.core.cv2.imread")
    def test_predict_rotation_improved_success(
        self, mock_imread, mock_osd, mock_reader_class, sample_image_bgr
    ):
        """Test successful rotation prediction."""
        from engineering_doc_parser.rotation.core import predict_rotation_improved

        # Setup mocks
        mock_imread.return_value = sample_image_bgr
        mock_osd.return_value = (0, 15.0)

        mock_reader = MagicMock()
        # Return good results for 0° (horizontal text)
        good_result = [
            ([[10, 20], [50, 20], [50, 40], [10, 40]], "Hello", 0.9),
            ([[60, 30], [120, 30], [120, 50], [60, 50]], "World", 0.85),
        ]
        mock_reader.readtext.return_value = good_result
        mock_reader_class.return_value = mock_reader

        angle, result, img, meta = predict_rotation_improved(
            image_path="dummy.png", lang_list=["en"], gpu=False
        )

        assert angle in [0, 90, 180, 270]
        assert img is not None
        assert "scores" in meta
        assert "num_detections" in meta
        assert len(meta["scores"]) == 4  # One score per angle

    @patch("engineering_doc_parser.rotation.core.easyocr.Reader")
    @patch("engineering_doc_parser.rotation.core.tesseract_osd_angle_conf")
    @patch("engineering_doc_parser.rotation.core.cv2.imread")
    def test_predict_rotation_improved_with_osd_shortlist(
        self, mock_imread, mock_osd, mock_reader_class, sample_image_bgr
    ):
        """Test rotation prediction with OSD shortlist enabled."""
        from engineering_doc_parser.rotation.core import predict_rotation_improved

        mock_imread.return_value = sample_image_bgr
        mock_osd.return_value = (90, 20.0)  # OSD suggests 90°

        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[10, 20], [50, 20], [50, 40], [10, 40]], "Text", 0.8),
        ]
        mock_reader_class.return_value = mock_reader

        angle, result, img, meta = predict_rotation_improved(
            image_path="dummy.png", lang_list=["en"], gpu=False, use_osd_shortlist=True
        )

        assert angle in [90, 270]  # Should be OSD angle or its 180° opposite
        assert "scores" in meta

    @patch("engineering_doc_parser.rotation.core.easyocr.Reader")
    @patch("engineering_doc_parser.rotation.core.tesseract_osd_angle_conf")
    @patch("engineering_doc_parser.rotation.core.cv2.imread")
    def test_predict_rotation_improved_enhance_contrast(
        self, mock_imread, mock_osd, mock_reader_class, sample_image_bgr
    ):
        """Test rotation prediction with contrast enhancement."""
        from engineering_doc_parser.rotation.core import predict_rotation_improved

        mock_imread.return_value = sample_image_bgr
        mock_osd.return_value = (None, None)

        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[10, 20], [50, 20], [50, 40], [10, 40]], "Text", 0.8),
        ]
        mock_reader_class.return_value = mock_reader

        # Mock cv2 operations for contrast enhancement
        with patch("engineering_doc_parser.rotation.core.cv2.cvtColor") as mock_cvt, patch(
            "engineering_doc_parser.rotation.core.cv2.createCLAHE"
        ) as mock_clahe:

            mock_cvt.side_effect = lambda img, code: img  # Return as-is
            mock_clahe_instance = MagicMock()
            mock_clahe_instance.apply.return_value = sample_image_bgr[:, :, 0]
            mock_clahe.return_value = mock_clahe_instance

            angle, result, img, meta = predict_rotation_improved(
                image_path="dummy.png",
                lang_list=["en"],
                gpu=False,
                enhance_contrast=True,
            )

            assert angle in [0, 90, 180, 270]

    @patch("engineering_doc_parser.rotation.core.easyocr.Reader")
    @patch("engineering_doc_parser.rotation.core.cv2.imread")
    def test_predict_rotation_improved_file_not_found(
        self, mock_imread, mock_reader_class
    ):
        """Test rotation prediction with missing file."""
        from engineering_doc_parser.rotation.core import predict_rotation_improved

        mock_imread.return_value = None
        mock_reader_class.return_value = MagicMock()

        with pytest.raises(FileNotFoundError):
            predict_rotation_improved(
                image_path="nonexistent.png", lang_list=["en"], gpu=False
            )


class TestAutorotateAndSave:
    """Tests for autorotate and save functions."""

    @patch("engineering_doc_parser.rotation.core.predict_rotation_improved")
    @patch("engineering_doc_parser.rotation.core.cv2.imwrite")
    def test_autorotate_and_save_improved_success(
        self, mock_imwrite, mock_predict, sample_image_bgr, tmp_path
    ):
        """Test successful autorotate and save."""
        from engineering_doc_parser.rotation.core import autorotate_and_save_improved

        mock_predict.return_value = (
            90,  # angle
            [],  # ocr_result
            sample_image_bgr,  # rotated_img
            {"scores": {0: 100, 90: 200, 180: 50, 270: 80}},  # meta
        )
        mock_imwrite.return_value = True

        output_path = tmp_path / "output.png"
        angle, ocr_result, saved_path, meta = autorotate_and_save_improved(
            image_path="input.png",
            output_path=str(output_path),
            lang_list=["en"],
            gpu=False,
            overwrite=True,
        )

        assert angle == 90
        assert saved_path == str(output_path)
        assert mock_imwrite.called

    @patch("engineering_doc_parser.rotation.core.predict_rotation_improved")
    def test_autorotate_and_save_improved_default_output(
        self, mock_predict, sample_image_bgr, tmp_path
    ):
        """Test autorotate with default output path."""
        from engineering_doc_parser.rotation.core import autorotate_and_save_improved

        mock_predict.return_value = (0, [], sample_image_bgr, {"scores": {}})

        with patch("engineering_doc_parser.rotation.core.cv2.imwrite", return_value=True):
            input_path = tmp_path / "document.png"
            angle, ocr_result, saved_path, meta = autorotate_and_save_improved(
                image_path=str(input_path),
                output_path=None,
                lang_list=["en"],
                gpu=False,
                overwrite=True,
            )

            # Should generate default name with rotation angle
            assert ".rot0.png" in saved_path or saved_path.endswith(".rot0.png")

    @patch("engineering_doc_parser.rotation.core.predict_rotation_improved")
    @patch("engineering_doc_parser.rotation.core.cv2.imwrite")
    def test_autorotate_and_save_improved_file_exists(
        self, mock_imwrite, mock_predict, sample_image_bgr, tmp_path
    ):
        """Test autorotate when output file exists without overwrite."""
        from engineering_doc_parser.rotation.core import autorotate_and_save_improved

        mock_predict.return_value = (0, [], sample_image_bgr, {})
        output_path = tmp_path / "output.png"
        output_path.write_bytes(b"existing")

        with pytest.raises(FileExistsError):
            autorotate_and_save_improved(
                image_path="input.png",
                output_path=str(output_path),
                lang_list=["en"],
                gpu=False,
                overwrite=False,
            )

    @patch("engineering_doc_parser.rotation.core.predict_rotation_improved")
    @patch("engineering_doc_parser.rotation.core.cv2.imwrite")
    def test_autorotate_and_save_improved_write_failure(
        self, mock_imwrite, mock_predict, sample_image_bgr
    ):
        """Test autorotate when image write fails."""
        from engineering_doc_parser.rotation.core import autorotate_and_save_improved

        mock_predict.return_value = (0, [], sample_image_bgr, {})
        mock_imwrite.return_value = False

        with pytest.raises(IOError):
            autorotate_and_save_improved(
                image_path="input.png",
                output_path="output.png",
                lang_list=["en"],
                gpu=False,
                overwrite=True,
            )


class TestAutorotateFolder:
    """Tests for folder batch processing."""

    @patch("engineering_doc_parser.rotation.core.predict_rotation_improved")
    @patch("engineering_doc_parser.rotation.core.cv2.imwrite")
    def test_autorotate_folder_success(
        self, mock_imwrite, mock_predict, tmp_path, sample_image_bgr
    ):
        """Test successful folder processing."""
        from engineering_doc_parser.rotation.core import autorotate_folder

        # Create test images
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "img1.png").write_bytes(b"dummy")
        (input_dir / "img2.jpg").write_bytes(b"dummy")
        (input_dir / "doc.txt").write_text("ignore")

        output_dir = tmp_path / "output"

        mock_predict.return_value = (
            0,
            [],
            sample_image_bgr,
            {
                "osd_angle": 0,
                "osd_conf": 15.0,
                "scores": {0: 100, 90: 50, 180: 30, 270: 60},
            },
        )
        mock_imwrite.return_value = True

        results = autorotate_folder(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            overwrite=True,
            lang_list=["en"],
            gpu=False,
        )

        assert len(results) == 2  # Two image files
        assert all(r[2] in [0, 90, 180, 270] for r in results)  # Valid angles
        assert output_dir.exists()

    @patch("engineering_doc_parser.rotation.core.predict_rotation_improved")
    def test_autorotate_folder_skip_existing(
        self, mock_predict, tmp_path, sample_image_bgr
    ):
        """Test folder processing skips existing files."""
        from engineering_doc_parser.rotation.core import autorotate_folder

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "img1.png").write_bytes(b"dummy")

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "img1.rot0.png").write_bytes(b"existing")

        mock_predict.return_value = (0, [], sample_image_bgr, {})

        results = autorotate_folder(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            overwrite=False,
            lang_list=["en"],
            gpu=False,
        )

        # Should skip existing file
        assert len(results) == 0

    @patch("engineering_doc_parser.rotation.core.predict_rotation_improved")
    def test_autorotate_folder_error_handling(
        self, mock_predict, tmp_path, sample_image_bgr
    ):
        """Test folder processing handles errors gracefully."""
        from engineering_doc_parser.rotation.core import autorotate_folder

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "img1.png").write_bytes(b"dummy")
        (input_dir / "img2.png").write_bytes(b"dummy")

        output_dir = tmp_path / "output"

        # First call succeeds, second raises exception
        mock_predict.side_effect = [
            (0, [], sample_image_bgr, {}),
            Exception("Processing error"),
        ]

        with patch("engineering_doc_parser.rotation.core.cv2.imwrite", return_value=True):
            results = autorotate_folder(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                overwrite=True,
                lang_list=["en"],
                gpu=False,
            )

            # Should process first image, skip second due to error
            assert len(results) == 1


class TestCLI:
    """Tests for CLI argument parsing."""

    def test_parse_args_defaults(self):
        """Test argument parser with defaults."""
        from engineering_doc_parser.rotation.core import parse_args

        with patch("sys.argv", ["new_rotation.py"]):
            args = parse_args()

            assert args.input_dir == "debug_crops_3408"
            assert args.output_dir == "out_folder_debug_crops_3408_rotated"
            assert args.lang == ["en"]
            assert args.gpu is False
            assert args.osd_weight == 3.0

    def test_parse_args_custom(self):
        """Test argument parser with custom values."""
        from engineering_doc_parser.rotation.core import parse_args

        with patch(
            "sys.argv",
            [
                "new_rotation.py",
                "--in",
                "custom_input",
                "--out",
                "custom_output",
                "--lang",
                "en",
                "fr",
                "--gpu",
                "--osd-weight",
                "5.0",
                "--use-osd-shortlist",
                "--overwrite",
            ],
        ):
            args = parse_args()

            assert args.input_dir == "custom_input"
            assert args.output_dir == "custom_output"
            assert args.lang == ["en", "fr"]
            assert args.gpu is True
            assert args.osd_weight == 5.0
            assert args.use_osd_shortlist is True
            assert args.overwrite is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
