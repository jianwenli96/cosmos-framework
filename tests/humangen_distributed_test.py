# SPDX-License-Identifier: OpenMDW-1.1
"""CPU checks for the HumanGen loader's distributed index stream."""

from itertools import islice

import pytest
import torch

from cosmos_framework.data.generator.joint_dataloader import RankPartitionedDataLoader
from cosmos_framework.utils.generator.parallelism import ParallelDims


class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {"index": index}


def loader(monkeypatch, rank, world, size, **kwargs):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    return RankPartitionedDataLoader(
        {"humangen": {"ratio": 1, "dataset": IndexDataset(size)}},
        distributed_shuffle=True,
        shuffle_seed=42,
        batch_size=1,
        **kwargs,
    )


@pytest.mark.parametrize("world,size", [(1, 17), (4, 20), (4, 23), (64, 131)])
def test_disjoint_balanced_reproducible_epochs(monkeypatch, world, size):
    epochs = [[], []]
    per_rank = size // world
    for rank in range(world):
        stream = iter(loader(monkeypatch, rank, world, size, num_workers=0))
        observed = [int(batch["index"][0]) for batch in islice(stream, 2 * per_rank)]
        again = iter(loader(monkeypatch, rank, world, size, num_workers=0))
        assert observed == [int(batch["index"][0]) for batch in islice(again, 2 * per_rank)]
        for epoch in range(2):
            epochs[epoch].extend(observed[epoch * per_rank : (epoch + 1) * per_rank])
    for values in epochs:
        assert len(values) == len(set(values)) == per_rank * world
        assert set(values) <= set(range(size))
        if size % world == 0:
            assert set(values) == set(range(size))
    assert epochs[0] != epochs[1]


def test_persistent_workers_cross_epoch(monkeypatch):
    dl = loader(monkeypatch, 0, 2, 14, num_workers=2, persistent_workers=True)
    reference = loader(monkeypatch, 0, 2, 14, num_workers=0)
    assert [int(x["index"][0]) for x in islice(iter(dl), 21)] == [
        int(x["index"][0]) for x in islice(iter(reference), 21)
    ]


def test_too_few_samples_fails_before_iteration(monkeypatch):
    with pytest.raises(ValueError, match="at least one sample"):
        loader(monkeypatch, 0, 4, 3)


def test_sampler_conflict_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="owns the sampler"):
        loader(monkeypatch, 0, 2, 8, shuffle=True)


@pytest.mark.parametrize("shard,replicas,expected", [(-1, 1, 64), (-1, 8, 8), (8, 8, 8), (-1, -1, 64)])
def test_parallel_topology(shard, replicas, expected):
    dims = ParallelDims(world_size=64, dp_shard=shard, dp_replicate=replicas)
    assert dims.dp_shard == expected
    assert dims.dp_shard * dims.dp_replicate == 64


def test_invalid_parallel_topology():
    with pytest.raises(ValueError):
        ParallelDims(world_size=64, dp_shard=-1, dp_replicate=3)
    with pytest.raises(ValueError):
        ParallelDims(world_size=64, dp_shard=8, dp_replicate=4)


@pytest.mark.parametrize("frame_ids", [[[2, 3, 5, 7, 9]], [[0, 1, 2, 3, 4], list(range(9))]])
def test_window_frame_metadata_survives_training_packing(frame_ids):
    from collections import deque
    from types import SimpleNamespace

    from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, custom_collate_fn

    samples = [dict(video=torch.zeros(3, len(ids), 2, 2), robot_frame_ids=ids) for ids in frame_ids]
    collated = custom_collate_fn(samples)
    loader = SimpleNamespace(
        buffers=[deque()],
        dataloaders=[iter([collated])],
        _MULTI_ITEM_KEYS=PackingDataLoader._MULTI_ITEM_KEYS,
        _FLATTEN_LIST_KEYS=PackingDataLoader._FLATTEN_LIST_KEYS,
    )
    packed = {}
    for ids in frame_ids:
        sample = PackingDataLoader._get_next_sample(loader, 0)
        assert sample["robot_frame_ids"] == ids
        PackingDataLoader._update_output_batch(loader, packed, sample)
    assert packed["robot_frame_ids"] == frame_ids
