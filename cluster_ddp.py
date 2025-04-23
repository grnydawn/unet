import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

def init_process_group():
    dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')

def get_cluster_group(num_clusters):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    assert world_size % num_clusters == 0, "World size must be divisible by number of clusters"
    cluster_size = world_size // num_clusters
    cluster_id = rank // cluster_size

    cluster_ranks = list(range(cluster_id * cluster_size, (cluster_id + 1) * cluster_size))

    group = dist.new_group(ranks=cluster_ranks)
    return group, cluster_id, rank % cluster_size, cluster_ranks

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )

    def forward(self, x):
        return self.net(x)

def main():

    # Set up SLURM-based distributed training environment
    os.environ['MASTER_ADDR'] = str(os.environ['HOSTNAME'])
    os.environ['MASTER_PORT'] = "29500"
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
    os.environ['RANK'] = os.environ['SLURM_PROCID']

    init_process_group()
    num_clusters = 2

    cluster_group, cluster_id, local_rank, cluster_ranks = get_cluster_group(num_clusters)

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    model = MyModel().to(device)
    ddp_model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None, process_group=cluster_group)

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # Dummy train loop
    for epoch in range(3):
        inputs = torch.randn(32, 10).to(device)
        targets = torch.randint(0, 2, (32,)).to(device)

        outputs = ddp_model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"[Cluster {cluster_id} | Rank {dist.get_rank()}] Epoch {epoch} Loss: {loss.item()}")

    dist.barrier(group=cluster_group)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()

