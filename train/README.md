# Table Detector (YOLO) — FOCR-6

> Branch name to use: **FOCR-6-Table-detector**

This repository preps a YOLO dataset and documents training for a table detector.

## Quickstart

```bash
# 1) Create and activate a virtual env (Python >= 3.10)
python -m venv .venv
source .venv/bin/activate   

# 2) Install deps
pip install -r requirements.txt

# 43) Prepare dataset (edit paths as needed or use CLI flags)
python scripts/prepare_dataset.py   --source-images "/code/Datasets/foolad/Data_unzipped/Data/3408/yolo_3408/3408_yolo/images"   --source-labels "/code/Datasets/foolad/Data_unzipped/Data/3408/yolo_3408/labels"   --dest-base "dataset_3408_1/dataset"   --train 0.7 --val 0.2 --seed 42
```

### YOLO training

Example `custom_data.yaml` is in `configs/custom_data.yaml`. Adjust the absolute paths for your environment.

```bash
yolo detect train   model=/code/Datasets/foolad/table_detect_ocr/runs_3407/exp13/weights/best.pt   data=configs/custom_data.yaml   epochs=1000 batch=8 workers=0 device=1   project=runs_3407_3408 name=exp1 plots=False
```

## Repo layout

```
.
├─ focr_table_detector/
│  ├─ __init__.py
│  └─ dataset.py
├─ scripts/
│  └─ prepare_dataset.py
├─ configs/
│  └─ custom_data.yaml
├─ tests/
│  ├─ test_dataset_split.py
├─ requirements.txt
```

## Git & Branching

```bash
git init
git checkout -b FOCR-6-Table-detector
git add .
git commit -m "feat(FOCR-6): bootstrap table detector repo skeleton"
# Create remote first at GitHub, then:
git remote add origin git@github.com:<org>/<repo>.git
git push -u origin FOCR-6-Table-detector
```

## Testing

```bash
pytest --cov=focr_table_detector  --cov-report=term-missing --cov-fail-under=85
```