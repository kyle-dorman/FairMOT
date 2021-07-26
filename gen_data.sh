#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "Must pass bucket name. e.g. `./gen_data.sh test-bucket`"
fi

if [ -z "$1" ]
  then
    echo "Must pass bucket name. e.g. `./gen_data.sh test-bucket`"
fi

BUCKET=$1

echo "Generating mot data"

python src/track.py mot --test_mot17 True --load_model models/fairmot_dla34.pth --conf_thres 0.4 --data_dir /app/data --exp_id fairmot_base

python src/track.py mot --val_mot17 True --load_model models/fairmot_dla34.pth --conf_thres 0.4 --data_dir /app/data --exp_id fairmot_base

python src/track.py mot --test_mot20 True --load_model models/fairmot_dla34.pth --conf_thres 0.3 --data_dir /app/data --exp_id fairmot_base

python src/track.py mot --val_mot20 True --load_model models/fairmot_dla34.pth --conf_thres 0.3 --data_dir /app/data --exp_id fairmot_base

echo "Done w/reid!"

gsutil -m rsync -d -r /app/data/experiments/fairmot_base gs://${BUCKET}/experiments/fairmot_base

python src/track.py mot --test_mot17 True --load_model models/fairmot_dla34.pth --conf_thres 0.4 --data_dir /app/data --exp_id fairmot_no_reid --skip-reid

python src/track.py mot --val_mot17 True --load_model models/fairmot_dla34.pth --conf_thres 0.4 --data_dir /app/data --exp_id fairmot_no_reid --skip-reid

python src/track.py mot --test_mot20 True --load_model models/fairmot_dla34.pth --conf_thres 0.3 --data_dir /app/data --exp_id fairmot_no_reid --skip-reid

python src/track.py mot --val_mot20 True --load_model models/fairmot_dla34.pth --conf_thres 0.3 --data_dir /app/data --exp_id fairmot_no_reid --skip-reid

echo "Done w/o reid!"

gsutil -m rsync -d -r /app/data/experiments/fairmot_no_reid gs://${BUCKET}/experiments/fairmot_no_reid

python src/extract_reid.py mot --data_dir /app/data --val_mot20 True --load_model models/fairmot_dla34.pth --exp_id reid_fairmot --conf_thres 0.05 --input_h 224 --input_w 96

python src/extract_reid.py mot --data_dir /app/data --val_mot17 True --load_model models/fairmot_dla34.pth --exp_id reid_fairmot --conf_thres 0.05 --input_h 224 --input_w 96

python src/extract_reid.py mot --data_dir /app/data --test_mot20 True --load_model models/fairmot_dla34.pth --exp_id reid_fairmot --conf_thres 0.05 --input_h 224 --input_w 96

python src/extract_reid.py mot --data_dir /app/data --test_mot17 True --load_model models/fairmot_dla34.pth --exp_id reid_fairmot --conf_thres 0.05 --input_h 224 --input_w 96

echo "Done generating reids"

gsutil -m rsync -r /app/data/experiments/gt_reid_fairmot gs://${BUCKET}/experiments/gt_reid_fairmot
gsutil -m rsync -r /app/data/experiments/FRCNN_reid_fairmot gs://${BUCKET}/experiments/FRCNN_reid_fairmot
gsutil -m rsync -r /app/data/experiments/SDP_reid_fairmot gs://${BUCKET}/experiments/SDP_reid_fairmot
gsutil -m rsync -r /app/data/experiments/DPM_reid_fairmot gs://${BUCKET}/experiments/DPM_reid_fairmot
