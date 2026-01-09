# YOLO Training, Fine-tuning, and Testing

These scripts use the Ultralytics API and your dataset described in `data.yaml`.

## Install

```powershell
# From the project folder
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you have a CUDA-capable GPU and want GPU acceleration, install a CUDA-specific PyTorch build BEFORE installing the rest:

```powershell
# Pick your CUDA version at https://pytorch.org/get-started/locally/
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Then install the remaining deps
pip install -r requirements.txt
```

### Low-disk/Container (CPU-only, smaller)

If your environment has limited disk (e.g., Pterodactyl) or you saw "No space left on device" while installing:

```bash
# Linux container example
export PIP_NO_CACHE_DIR=1
pip cache purge
pip install --no-cache-dir -r requirements-cpu.txt
```

This uses CPU-only wheels and headless OpenCV to reduce download size. You can enable plotting later by installing `matplotlib`.

## Train a model (без параметров)

Отредактируйте `configs/train.yaml`, затем запустите:

```powershell
python train.py
```
Артефакты будут сохранены в `runs/train`.

### Советы для лучшего качества
- Модель: по умолчанию стоит `yolo11n.pt` (быстро, но базовое качество). Для лучше качества используйте `yolo11s.pt` или `yolo11m.pt` в `configs/train.yaml` (медленнее, но точнее).
- Эпохи: увеличьте `epochs` до 100–200, если набор данных небольшой/средний и время позволяет.
- Размер: `imgsz: 640` — баланс; `768/960` могут поднять качество, но потребуют больше ресурсов.
- Аугментации: параметры уже настроены (mosaic, hsv, mixup и т.п.). Если данные очень однородные — уменьшите силу аугментаций; если разнообразные — можно оставить/усилить.
- Схема обучения: включены AdamW, cosine LR, EMA — это повышает стабильность и итоговую метрику.
- CPU/без GPU: скрипт автоматически переключит `device` на `cpu` при отсутствии CUDA.

## Fine-tune from existing weights (без параметров)

Отредактируйте `configs/finetune.yaml`, затем запустите:

```powershell
python finetune.py
```
Артефакты будут сохранены в `runs/finetune`.

Рекомендации при дообучении:
- Сниженный `lr0` и мягче аугментации уже заданы в `configs/finetune.yaml`.
- При дообучении на близком по распределению датасете можно заморозить ранние слои: `freeze: 10` (или 0, если хотите дообучить всё).

## Test/Validate a trained model (без параметров)

Отредактируйте `configs/test.yaml`, затем запустите:

```powershell
python test.py
```
Укажите `split: val` в конфиге для оценки на валидации. Метрики и графики будут сохранены в `runs/val`.

## Notes
- Ensure `data.yaml` correctly points to your `train/`, `val/`, and `test/` folders.
- If you see an import error for `ultralytics`, run `pip install -r requirements.txt`.
