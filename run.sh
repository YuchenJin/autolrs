dir=log
mkdir -p $dir

MIN_LR=0.001
MAX_LR=0.1
PORT=12315

python autolrs_server.py --min_lr $MIN_LR --max_lr $MAX_LR --port $PORT > $dir/server_log 2>&1 & echo $!

CUDA_VISIBLE_DEVICES=0 python cifar10_example.py --port ${PORT}> $dir/client_log 2>&1
