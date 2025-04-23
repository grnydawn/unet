print("start", flush=True)
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

print("module loaded", flush=True)

def setup():
    dist.init_process_group(backend="nccl", init_method="env://")

def cleanup():
    dist.destroy_process_group()

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

def main():
    print("TP 0", flush=True)
    # Get environment variables set by SLURM
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ["SLURM_LOCALID"])  # Local GPU ID on the node

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ['MASTER_ADDR'] = str(os.environ['HOSTNAME'])
    #os.environ['MASTER_ADDR'] = "frontier00255"
    os.environ['MASTER_PORT'] = "29502"

    print("TP 1", flush=True)
    # Set the correct CUDA device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    print("TP 2", flush=True)
    setup()

    print("TP 3", flush=True)
    # Set up model on device
    model = ToyModel().to(device)
    ddp_model = DDP(model, device_ids=[local_rank])

    print("TP 4", flush=True)
    # Define subgroups (you can also automate this)
    group1_ranks = [0, 1]
    group2_ranks = [2, 3]

    group1 = dist.new_group(ranks=group1_ranks)
    group2 = dist.new_group(ranks=group2_ranks)

    print("TP 5", flush=True)
    data = torch.ones(10, 10, device=device) * (rank + 1)

    print("TP 6", flush=True)
    if rank in group1_ranks:
        dist.all_reduce(data, group=group1)
        print(f"Rank {rank} in group 1: {data}", flush=True)
    elif rank in group2_ranks:
        dist.all_reduce(data, group=group2)
        print(f"Rank {rank} in group 2: {data}", flush=True)

    print("TP 7", flush=True)
    cleanup()

    print("TP 8", flush=True)
if __name__ == "__main__":
    print("main", flush=True)
    main()

