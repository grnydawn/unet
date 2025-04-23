# File: trainer.py
import os
import gc
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from data_loader import load_bias_correction_data
from utils import timed_event


def train_epoch(model, loader, criterion, optimizer, scaler, device, logger):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        timed_event(logger, "batch_train")
    return total_loss / len(loader)


def validate_epoch(model, loader, criterion, device, logger):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast():
                out = model(x)
                loss = criterion(out, y)
            val_loss += loss.item()
            timed_event(logger, "batch_val")
    return val_loss / len(loader)


def train_loop(
    model, train_ds, val_ds,
    criterion, optimizer, outdir: str,
    epochs: int, batch_size: int,
    patience: int, device, local_rank: int, world_size: int,
    logger
) -> dict:
    # Setup DDP
    train_sampler = DistributedSampler(train_ds, world_size, local_rank)
    val_sampler = DistributedSampler(val_ds, world_size, local_rank, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size, sampler=val_sampler)

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])
    scaler = torch.cuda.amp.GradScaler()

    best_loss = float('inf')
    epochs_no_improve = 0
    history = {'train': [], 'val': []}

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, logger)
        val_loss = validate_epoch(model, val_loader, criterion, device, logger)
        if local_rank == 0:
            logger.info(f"Epoch {epoch}: train={tr_loss:.4f}, val={val_loss:.4f}")

        history['train'].append(tr_loss)
        history['val'].append(val_loss)

        # Checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            if local_rank == 0:
                torch.save(model.module.state_dict(), os.path.join(outdir, 'best.pth'))
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

        dist.barrier()
        gc.collect()
        torch.cuda.empty_cache()

    return {'best_loss': best_loss, **history}


# File: main.py
import os
import argparse
import torch
import torch.distributed as dist
from utils import setup_logging, seed_everything, timed_event
from dataset import BiasCorrectionDataset
from metrics import calculate_metrics
from trainer import train_loop
from model_hub import UNet, ResidualUNet, AttentionUNet, UNetPlusPlus, ResidualUNetPlusPlus

MODEL_MAP = {
    'unet': UNet,
    'residual_unet': ResidualUNet,
    'attention_unet': AttentionUNet,
    'unet_plus_plus': UNetPlusPlus,
    'residual_unet_plus': ResidualUNetPlusPlus,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODEL_MAP.keys(), default='unet', help="Choose a model")
    parser.add_argument('--epochs', type=int, default=100, help="Specify the number of epochs")
    parser.add_argument("--dataset", type=str, default="ResidualUNet", help="Specify the dataset name")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size for the model")
    parser.add_argument('--patience', type=int, default=5, help="early exits criteria")
    parser.add_argument('--lr', type=float, default=5e-4, , help="learning rate")
    parser.add_argument('--outdir', type=str, default='.', help="output directory")
    return parser.parse_args()


def init_distributed():
    dist.init_process_group('nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    logger = setup_logging('training', args.outdir)
    seed_everything(42)
    timed_event(logger, 'start')

    local_rank, world_size = init_distributed()

    # Load data and stats ... (omitted: use np.load similarly)
    train_data, val_data, input_stats, target_stats = load_bias_correction_data(
        input_base_dir, target_base_dir, lead_time,
        input_vars, output_vars, test=False
    )

    train_ds = BiasCorrectionDataset(**train_data, input_stats=input_stats, target_stats=target_stats)
    val_ds = BiasCorrectionDataset(**val_data, input_stats=input_stats, target_stats=target_stats)

    model = MODEL_MAP[args.model](in_channels=len(input_stats), out_channels=len(target_stats))
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    results = train_loop(
        model, train_ds, val_ds, criterion, optimizer,
        args.outdir, args.epochs, args.batch_size,
        args.patience, torch.device(f'cuda:{local_rank}'),
        local_rank, world_size, logger
    )

    if local_rank == 0:
        # Load best model and evaluate
        model.load_state_dict(torch.load(os.path.join(args.outdir, 'best.pth')))
        # for test
        test_data = load_bias_correction_data(
            input_base_dir, target_base_dir, lead_time,
            input_vars, output_vars, test=True
        )
        test_ds = BiasCorrectionDataset(**test_data, input_stats=input_stats, target_stats=target_stats)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size)
  
        truths, preds = evaluate_model(model, test_loader, torch.device(f'cuda:{local_rank}'))
        metrics = calculate_metrics(truths, preds)
        logger.info(f"Test metrics: {metrics}")

    timed_event(logger, 'end')

if __name__ == '__main__':
    main()

