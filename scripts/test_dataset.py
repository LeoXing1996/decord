import os
import torch
import os.path as osp
from copy import deepcopy
import time

import torch.distributed as dist
from accelerate import PartialState
import decord
from decord import VideoReader, gpu
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import subprocess

from argparse import ArgumentParser


def init_dist(launcher="slurm", backend="nccl", port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == "pytorch":
        rank = int(os.environ["RANK"])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)

    elif launcher == "slurm":
        proc_id = int(os.environ["SLURM_PROCID"])
        ntasks = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(ntasks)
        os.environ["RANK"] = str(proc_id)
        port = os.environ.get("PORT", port)
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(backend=backend)
        print(
            f"proc_id: {proc_id}; local_rank: {local_rank}; ntasks: {ntasks}; "
            f"node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}"
        )

    else:
        raise NotImplementedError(f"Not implemented launcher type: `{launcher}`!")

    return local_rank


class AnimeDataset(IterableDataset):
    def __init__(
        self,
        video_path: str,
        frames_at_once: int,
        rank: int,
        world_size: int,
    ):
        super().__init__()
        decord.bridge.set_bridge('torch')

        self.file_iterator = self.get_file_list_iter(video_path)

        self.frames_at_once = frames_at_once
        self.rank = rank
        self.world_size = world_size

    def get_file_list_iter(self, video_path):
        class file_iterator:
            def __iter__(self):
                for f in os.listdir(video_path):
                    if f.endswith(".mp4"):
                        yield osp.join(video_path, f)

        return file_iterator()

    def __iter__(self):
        worker_info = get_worker_info()
        mod = self.world_size
        shift = self.rank
        if worker_info is not None:
            mod *= worker_info.num_workers
            shift = self.rank * worker_info.num_workers + worker_info.id

        dataset_meta = {
            "rank": self.rank,
            "world_size": self.world_size,
            "num_worker": worker_info.num_workers if worker_info is not None else 0,
            "worker_id": worker_info.id if worker_info is not None else 0,
            "mod": mod,
            "shift": shift,
        }

        for idx, file in enumerate(self.file_iterator):
            if (idx + shift) % mod == 0:
                with open(file, "rb") as file_io:
                    reader = VideoReader(file_io, ctx=gpu(self.rank))
                    for start_idx in range(0, len(reader), self.frames_at_once):
                        end_idx = min(start_idx + self.frames_at_once, len(reader))
                        try:
                            frames = reader.get_batch(
                                list(range(start_idx, end_idx))
                            )
                            is_error = False
                        except Exception:
                            print(f'Error at {file} ({start_idx}: {end_idx})')
                            continue

                        yield {
                            "frame": frames,
                            "video_name": [file],
                            "start_idx": start_idx,
                            "end_idx": end_idx,
                            "dataset_meta": deepcopy(dataset_meta),
                            "is_error": is_error,
                        }

    @classmethod
    def build_dataset(cls, video_path: str, frames_at_once: int = 500):
        state = PartialState()
        if state.use_distributed:
            rank, world_size = dist.get_rank(), dist.get_world_size()
        else:
            rank, world_size = 0, 1

        return cls(video_path, frames_at_once, rank, world_size)


def main(args):

    init_dist()

    data_root = args.data_root
    dataset = AnimeDataset.build_dataset(data_root, args.frames_at_once)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    rank, world_size = dist.get_rank(), dist.get_world_size()
    s = time.time()

    for data in dataloader:
        frame = data.pop('frame')
        with open(f'r{rank}-w{world_size}.txt', 'a+') as f:
            f.write(str(frame.shape) + '\n')
            f.write(str(data) + '\n')

    e = time.time()
    print(f"Total Time: {e - s}")


if __name__ == '__main__':
    """
    GPUS=4 GPUS_PER_NODE=4 bash test.sh test_dataset
    """
    parser = ArgumentParser()
    parser.add_argument('--data-root', type=str, default='anime_cases')
    parser.add_argument('--frames-at-once', type=int, default=10)
    args = parser.parse_args()

    main(args)