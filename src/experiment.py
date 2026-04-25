import csv
import json
import shutil
from datetime import datetime
from pathlib import Path


def save_metrics(exp_dir: Path, metrics: dict) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    path = exp_dir / 'metrics.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f'✅ metrics saved: {path}')


def snapshot_config(exp_dir: Path, config_path: Path) -> None:
    """Copy the current config.yaml into the experiment directory."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, exp_dir / 'config_snapshot.yaml')
    print(f'✅ config snapshot saved: {exp_dir / "config_snapshot.yaml"}')


def log_to_csv(experiments_dir: Path, exp_id: str, exp_name: str,
               train_metrics: dict, val_metrics: dict,
               kaggle_score: float = None, note: str = '') -> None:
    """
    將單次實驗結果 append 到 experiments/results.csv。
    若 CSV 不存在則自動建立並寫入 header。
    若該 exp_id 已存在則更新該行（避免重複）。
    """
    csv_path = experiments_dir / 'results.csv'
    fieldnames = [
        'experiment_id', 'name', 'date',
        'train_accuracy', 'train_precision', 'train_recall', 'train_f1',
        'val_accuracy',   'val_precision',   'val_recall',   'val_f1',
        'kaggle_score', 'note',
    ]
    new_row = {
        'experiment_id':   exp_id,
        'name':            exp_name,
        'date':            datetime.now().strftime('%Y-%m-%d %H:%M'),
        'train_accuracy':  train_metrics.get('accuracy',  ''),
        'train_precision': train_metrics.get('precision', ''),
        'train_recall':    train_metrics.get('recall',    ''),
        'train_f1':        train_metrics.get('f1',        ''),
        'val_accuracy':    val_metrics.get('accuracy',    ''),
        'val_precision':   val_metrics.get('precision',   ''),
        'val_recall':      val_metrics.get('recall',      ''),
        'val_f1':          val_metrics.get('f1',          ''),
        'kaggle_score':    kaggle_score if kaggle_score is not None else '',
        'note':            note,
    }

    rows = []
    if csv_path.exists():
        with open(csv_path, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        # 若已存在相同 exp_id，更新該行
        rows = [r for r in rows if r['experiment_id'] != exp_id]

    rows.append(new_row)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'✅ results.csv updated: {csv_path}')
