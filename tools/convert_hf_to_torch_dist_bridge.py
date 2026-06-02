import gc
import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core.enums import ModelType
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, save_checkpoint
from megatron.training.training import get_model

import slime_plugins.megatron_bridge  # noqa: F401
from slime.backends.megatron_utils.arguments import set_default_megatron_args
from slime.backends.megatron_utils.initialize import init
from slime.backends.megatron_utils.model_provider import get_model_provider_func
from slime.utils import megatron_bridge_utils
from slime.utils.logging_utils import configure_logger
from slime.utils.memory_utils import print_memory


def add_conversion_args(parser):
    parser.add_argument("--hf-checkpoint", type=str, required=True, help="HuggingFace model path")
    parser.add_argument(
        "--megatron-to-hf-mode",
        choices=["bridge"],
        default="bridge",
        help="Use megatron.bridge for HuggingFace -> Megatron conversion.",
    )
    try:
        parser.add_argument("--padded-vocab-size", type=int, default=None)
    except Exception:
        pass
    return parser


def get_args():
    args = parse_args(add_conversion_args)
    args = set_default_megatron_args(args)

    args.save_interval = 1
    args.micro_batch_size = 1
    args.global_batch_size = int(os.environ.get("WORLD_SIZE", "1"))

    validate_args(args)
    return args


def _init_distributed():
    world_size = int(os.getenv("WORLD_SIZE") or os.getenv("SLURM_NTASKS") or 1)
    local_rank = int(os.getenv("LOCAL_RANK") or os.getenv("SLURM_LOCALID") or 0)
    global_rank = int(os.getenv("RANK") or os.getenv("SLURM_PROCID") or 0)

    torch.cuda.set_device(local_rank)
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("RANK", str(global_rank))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=global_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )


def _mark_release_checkpoint(save_dir: str):
    tracker_filename = get_checkpoint_tracker_filename(save_dir)
    with open(tracker_filename, "w") as f:
        f.write("release")

    source_dir = Path(get_checkpoint_name(save_dir, 1, False, return_base_dir=True))
    target_dir = Path(get_checkpoint_name(save_dir, -1, True, return_base_dir=True))
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.move(source_dir, target_dir)


def main():
    configure_logger()
    _init_distributed()

    args = get_args()
    init(args)

    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)

    from megatron.bridge import AutoBridge

    hf_model_path = args.hf_checkpoint
    bridge = megatron_bridge_utils.patch_auto_bridge_hf_config(
        AutoBridge.from_hf_pretrained(hf_model_path, trust_remote_code=True)
    )
    with megatron_bridge_utils.patch_megatron_model(model):
        bridge.load_hf_weights(model)

    if args.use_cpu_initialization:
        model[0] = model[0].cpu()

    print_memory("after loading HF weights with megatron.bridge")
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    save_checkpoint(1, model, None, None, 0)

    if dist.get_rank() == 0:
        _mark_release_checkpoint(args.save)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
