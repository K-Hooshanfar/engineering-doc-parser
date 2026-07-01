# Table Detector (YOLO)

Prepare YOLO datasets and document training for a table detector.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python scripts/prepare_dataset.py \
  --source-images /path/to/images \
  --source-labels /path/to/labels \
  --dest-base dataset/output \
  --train 0.7 --val 0.2 --seed 42
```

### YOLO training

Example `custom_data.yaml` is in `configs/custom_data.yaml`. Adjust paths for your environment.

```bash
yolo detect train \
  model=/path/to/best.pt \
  data=configs/custom_data.yaml \
  epochs=1000 batch=8 workers=0 device=0
```

## Layout

```
src/engineering_doc_parser/table_detector/
  __init__.py
  dataset.py
scripts/prepare_dataset.py
configs/custom_data.yaml
```

## Testing

```bash
pytest --cov=engineering_doc_parser.table_detector --cov-report=term-missing
```
