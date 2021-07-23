#!/bin/bash

echo "Generating mot data"

python src/track.py mot --test_mot17 True --load_model models/fairmot_dla34.pth --conf_thres 0.4 --data_dir /app/data --exp_id fairmot_base

python src/track.py mot --val_mot17 True --load_model models/fairmot_dla34.pth --conf_thres 0.4 --data_dir /app/data --exp_id fairmot_base

python src/track.py mot --test_mot20 True --load_model models/fairmot_dla34.pth --conf_thres 0.3 --data_dir /app/data --exp_id fairmot_base

python src/track.py mot --val_mot20 True --load_model models/fairmot_dla34.pth --conf_thres 0.3 --data_dir /app/data --exp_id fairmot_base

echo "Done w/reid!"

gsutil -m rsync /app/data/experiments/fairmot_base gs://reid-tracking-fix/experiments/fairmot_base

python src/track.py mot --test_mot17 True --load_model models/fairmot_dla34.pth --conf_thres 0.4 --data_dir /app/data --exp_id fairmot_no_reid --skip-reid

python src/track.py mot --val_mot17 True --load_model models/fairmot_dla34.pth --conf_thres 0.4 --data_dir /app/data --exp_id fairmot_no_reid --skip-reid

python src/track.py mot --test_mot20 True --load_model models/fairmot_dla34.pth --conf_thres 0.3 --data_dir /app/data --exp_id fairmot_no_reid --skip-reid

python src/track.py mot --val_mot20 True --load_model models/fairmot_dla34.pth --conf_thres 0.3 --data_dir /app/data --exp_id fairmot_no_reid --skip-reid

echo "Done w/o reid!"

gsutil -m rsync /app/data/experiments/fairmot_no_reid gs://reid-tracking-fix/experiments/fairmot_no_reid

python src/extract_reid.py mot --data_dir /app/data --val_mot20 True --load_model models/fairmot_dla34.pth --exp_id gt_det_reid_fairmot --conf_thres 0.05

python src/extract_reid.py mot --data_dir /app/data --val_mot17 True --load_model models/fairmot_dla34.pth --exp_id gt_det_reid_fairmot --conf_thres 0.05

python src/extract_reid.py mot --data_dir /app/data --test_mot20 True --load_model models/fairmot_dla34.pth --exp_id gt_det_reid_fairmot --conf_thres 0.05

python src/extract_reid.py mot --data_dir /app/data --test_mot17 True --load_model models/fairmot_dla34.pth --exp_id gt_det_reid_fairmot --conf_thres 0.05

echo "Done generating reids"

gsutil -m rsync /app/data/experiments/gt_det_reid_fairmot gs://reid-tracking-fix/experiments/gt_det_reid_fairmot
