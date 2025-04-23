import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader
from datetime import timedelta

def event(msg):
    if os.environ["SLURM_PROCID"] == "0":
        print(msg, flush=True)

# Common model for all subgroups
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

# Create different datasets per group
def get_dataset(group_id):
    torch.manual_seed(42 + group_id)  # Different seed per group
    x = torch.randn(100, 10)
    y = torch.randint(0, 5, (100,))
    return TensorDataset(x, y)

def main():
    event("begin main")

    # SLURM environment variables
    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    event(f"MASTER_ADDR = {os.environ['MASTER_ADDR']}")
    event(f"MASTER_PORT = {os.environ['MASTER_PORT']}")

    event(f"WORLD_SIZE = {world_size}")
    event(f"RANK = {rank}")
    event(f"LOCAL_RANK = {local_rank}")

    # Set the device for each rank
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)

    #print(f"rank = {rank}, num_gpus = {num_gpus}, set_device({local_rank % num_gpus})", flush=True)

    event("set the device for each rank")

#        world_size=world_size,
#        rank=rank,

    # Initialize default process group
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=120)
    )

    event("Initialize default process group")

    # Setup subgroups
    num_subgroups = 8
    ranks_per_group = world_size // num_subgroups
    group_id = rank // ranks_per_group
    group_ranks = list(range(group_id * ranks_per_group, (group_id + 1) * ranks_per_group))
    subgroup = dist.new_group(
        ranks=group_ranks,
        use_local_synchronization=True,
        backend="nccl"
    )

    print(f"rank = {rank}, ranks_per_group = {ranks_per_group}, group_ranks = {group_ranks}", flush=True)
    event("Setup 8 subgroups")

    # Shared model and optimizer
    model = MyModel().cuda()
    ddp_model = DDP(model, device_ids=[local_rank], process_group=subgroup)
    event("Shared model and optimizer")

    # Group-specific dataset
    dataset = get_dataset(group_id)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.01)
    event("Group-specific dataset")

    # Training loop
    for epoch in range(3):
        event(f"Training epoch {epoch}")
        for x, y in dataloader:
            event(f"training data is ready")
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            output = ddp_model(x)
            event(f"forward")
            loss = criterion(output, y)
            loss.backward()
            event(f"backword")
            optimizer.step()
            event(f"optimize")
        if rank in group_ranks:
            print(f"[Rank {rank}] Group {group_id} - Epoch {epoch} - Loss: {loss.item():.4f}", flush=True)

    event("finish training")
    dist.destroy_process_group()

    event("finish main")

if __name__ == "__main__":
    main()
