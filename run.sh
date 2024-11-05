# run cora amazon-photos and amazon-computers weighted and not
conda activate graph
python3 train.py --flagfile=./config/cora.cfg
python3 train.py --flagfile=./config/amazon-photos.cfg
python3 train.py --flagfile=./config/amazon-computers.cfg
python3 train.py --flagfile=./config/cora_weighted.cfg
python3 train.py --flagfile=./config/amazon-photos_weighted.cfg
python3 train.py --flagfile=./config/amazon-computers_weighted.cfg