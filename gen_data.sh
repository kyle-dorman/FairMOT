#!/bin/bash

echo "Generating mot data"

python src/track.py mot --test_mot17 True --load_model models/fairmot_dla34.pth --conf_thres 0.4 --data_dir /app/data --exp_id base

python src/track.py mot --val_mot17 True --load_model models/fairmot_dla34.pth --conf_thres 0.4 --data_dir /app/data --exp_id base

python src/track.py mot --test_mot20 True --load_model models/fairmot_dla34.pth --conf_thres 0.3 --data_dir /app/data --exp_id base

python src/track.py mot --val_mot20 True --load_model models/fairmot_dla34.pth --conf_thres 0.3 --data_dir /app/data --exp_id base

echo "Done w/reid!"

gsutil -m rsync /app/data/results/base gs://kyle-reid-tracking-fix/fairmot_base

python src/track.py mot --test_mot17 True --load_model models/fairmot_dla34.pth --conf_thres 0.4 --data_dir /app/data --exp_id no_reid --skip-reid

python src/track.py mot --val_mot17 True --load_model models/fairmot_dla34.pth --conf_thres 0.4 --data_dir /app/data --exp_id no_reid --skip-reid

python src/track.py mot --test_mot20 True --load_model models/fairmot_dla34.pth --conf_thres 0.3 --data_dir /app/data --exp_id no_reid --skip-reid

python src/track.py mot --val_mot20 True --load_model models/fairmot_dla34.pth --conf_thres 0.3 --data_dir /app/data --exp_id no_reid --skip-reid

echo "Done w/o reid!"

gsutil -m rsync /app/data/results/no_reid gs://kyle-reid-tracking-fix/fairmot_no_reid
