"""Qwen2.5-VL document field extraction from cropped images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from tqdm import tqdm

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "prompts"
    / "document_extraction.txt"
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class ExtractionConfig:
    model_id: str = DEFAULT_MODEL_ID
    device: str = "cuda"
    max_new_tokens: int = 128
    flash_attention_2: bool = False
    prompt_path: Optional[Path] = None
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None
    glob_pattern: str = "*.crop.png"
    verbose: bool = True


def load_prompt(prompt_path: Optional[Path] = None) -> str:
    path = prompt_path or DEFAULT_PROMPT_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def collect_images(directory: Path, glob_pattern: str) -> List[Path]:
    directory = Path(directory)
    files = sorted(directory.glob(glob_pattern))
    if files:
        return [p for p in files if p.is_file()]

    # Fallback: any supported image in the folder.
    found: List[Path] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            found.append(path)
    return found


class QwenVLExtractor:
    """Run Qwen2.5-VL extraction on document crop images."""

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self._model = None
        self._processor = None
        self._prompt = load_prompt(config.prompt_path)

    def load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto",
        }
        if self.config.flash_attention_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            model_kwargs["torch_dtype"] = torch.bfloat16

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model_id,
            **model_kwargs,
        )

        processor_kwargs = {}
        if self.config.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.config.min_pixels
        if self.config.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.config.max_pixels

        self._processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            **processor_kwargs,
        )

    def extract(self, image_path: Path) -> str:
        from qwen_vl_utils import process_vision_info

        if self._model is None or self._processor is None:
            self.load()

        image_path = Path(image_path).resolve()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": self._prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.config.device)

        generated_ids = self._model.generate(
            **inputs, max_new_tokens=self.config.max_new_tokens
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip()

    def extract_and_save(
        self, image_path: Path, output_path: Optional[Path] = None
    ) -> Path:
        result = self.extract(image_path)
        md_path = output_path or Path(image_path).with_suffix(".md")
        md_path.write_text(result + "\n", encoding="utf-8")
        return md_path


def extraction_config_from_dict(data: dict) -> ExtractionConfig:
    prompt_path = data.get("prompt_path")
    return ExtractionConfig(
        model_id=str(data.get("model_id", DEFAULT_MODEL_ID)),
        device=str(data.get("device", "cuda")),
        max_new_tokens=int(data.get("max_new_tokens", 128)),
        flash_attention_2=bool(data.get("flash_attention_2", False)),
        prompt_path=Path(prompt_path) if prompt_path else None,
        min_pixels=data.get("min_pixels"),
        max_pixels=data.get("max_pixels"),
        glob_pattern=str(data.get("glob_pattern", "*.crop.png")),
        verbose=bool(data.get("verbose", True)),
    )


def extract_from_directory(
    input_dir: str | Path,
    config: Optional[ExtractionConfig] = None,
    *,
    config_dict: Optional[dict] = None,
) -> Tuple[int, int, List[Path]]:
    """Extract fields from all crop images in a directory. Returns (saved, failed, paths)."""
    cfg = config or extraction_config_from_dict(config_dict or {})
    directory = Path(input_dir)
    if not directory.is_dir():
        raise ValueError(f"Input directory not found: {directory}")

    images = collect_images(directory, cfg.glob_pattern)
    if not images:
        raise ValueError(
            f"No images found in {directory} (pattern: {cfg.glob_pattern})"
        )

    extractor = QwenVLExtractor(cfg)
    if cfg.verbose:
        print(f"Loading {cfg.model_id} ...")
    extractor.load()

    saved = 0
    failed = 0
    written: List[Path] = []

    iterator: Sequence[Path] = images
    if cfg.verbose:
        iterator = tqdm(images, desc="Qwen extraction", unit="img")

    for image_path in iterator:
        try:
            if cfg.verbose and not isinstance(iterator, tqdm):
                print(f"Processing: {image_path.name}")
            md_path = extractor.extract_and_save(image_path)
            saved += 1
            written.append(md_path)
            if cfg.verbose:
                msg = f"Saved: {md_path}"
                if isinstance(iterator, tqdm):
                    tqdm.write(msg)
                else:
                    print(msg)
        except Exception as exc:
            failed += 1
            msg = f"Failed: {image_path.name} | {exc}"
            if isinstance(iterator, tqdm):
                tqdm.write(msg)
            elif cfg.verbose:
                print(msg)

    if cfg.verbose:
        print(
            f"\nExtraction complete: {saved} saved, {failed} failed, {len(images)} total."
        )

    return saved, failed, written
