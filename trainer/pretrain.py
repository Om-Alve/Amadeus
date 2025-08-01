import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer
from model.model import AmadeusConfig, AmadeusForCausalLM
from dataset.dataset import PretrainDataset
from torch.utils.tensorboard.writer import SummaryWriter

warnings.filterwarnings('ignore')

def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)

def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, writer):
    loss_fnc = nn.CrossEntropyLoss(reduction="none")
    start_time = time.time()
    for step, (X,Y,loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iters_per_epoch + step, args.epochs * iters_per_epoch, args.learning_rate)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with ctx:
            res = model(X)
            loss = loss_fnc(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1),
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 and step != 0:
            spent_time = time.time() - start_time
            Logger(
    f"Epoch: {epoch + 1} / {args.epochs} | Step: {step} / {iters_per_epoch} | Loss: {(loss.item() * args.accumulation_steps):.4f} | LR: {lr:.6f} | Time: {spent_time:.2f}s"
            )
            if (writer is not None) and (not ddp or dist.get_rank() == 0):
                global_step = epoch * iters_per_epoch + step
                writer.add_scalar("loss", loss.item() * args.accumulation_steps, global_step)
                writer.add_scalar("lr", lr, global_step)
            start_time = time.time()

        if step % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            ckp = f"{args.save_dir}/pretrain_{lm_config.hidden_size}-2.pth"
            if isinstance(model, DDP):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)

def init_model(lm_config: AmadeusConfig):
    tokenizer = AutoTokenizer.from_pretrained("../model/")
    model = AmadeusForCausalLM(lm_config).to(args.device)
    Logger(f"Model loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    Logger(f"Tokens per iteration: {tokens_per_iter}")
    return model, tokenizer

def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(ddp_local_rank)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain Amadeus")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_tb", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()

    lm_config = AmadeusConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK", 1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"

    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        torch.cuda.manual_seed(base_seed + rank)

    if args.use_tb and (not ddp or dist.get_rank() == 0):
        os.makedirs("../tensorboard-logs", exist_ok=True)
        writer = SummaryWriter(log_dir="../tensorboard-logs")
    else:
        writer = None

    model, tokenizer = init_model(lm_config)
    model = torch.compile(model)
    train_ds = PretrainDataset(args.data_path, tokenizer, args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DDP(model, device_ids=[ddp_local_rank])

    iters_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, writer)

