# python -m torch.distributed.launch --nproc_per_node 8 train_ScanNet_guided_2cm_DDPV2.py # --config ./configWenxuanGuide2cmDDPL5.yaml
# python -m torch.distributed.launch --nproc_per_node 8 train_ScanNet_guided_2cm_DDP_WarmUP_FT.py --config ./configWenxuanGuide2cmDDPL5WarmUP.yaml
python -m torch.distributed.launch --nproc_per_node 8 train_ScanNet_guided_2cm_DDP_WarmUP.py --config ./configWenxuanGuide2cmDDPL5WarmUP.yaml
# python -m torch.distributed.launch --nproc_per_node 8 train_ScanNet_guided_2cm_DDP_WarmUP.py --config ./configWenxuanGuide2cmDDPL6WarmUP.yaml