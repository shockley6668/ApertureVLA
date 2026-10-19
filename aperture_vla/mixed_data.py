"""Shuffle encoded episodes and interleave suites before image decoding."""
from io import BytesIO

import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds

from turbovla.data.mixed_suite import LiberoMixedRLDSDataset


class SafeMixedRLDSDataset(LiberoMixedRLDSDataset):
    def _iter_mixed_episodes(self, epoch_seed, rank, world_size, worker_info):
        streams = []
        for path in self.dataset_dirs:
            builder = tfds.builder_from_directory(builder_dir=path)
            options = tf.data.Options()
            options.threading.private_threadpool_size = 2
            options.threading.max_intra_op_parallelism = 1
            options.autotune.enabled = False
            dataset = builder.as_dataset(
                split=self.split,
                decoders={"steps": {"observation": {
                    "image": tfds.decode.SkipDecoding(),
                    "wrist_image": tfds.decode.SkipDecoding(),
                }}},
                read_config=tfds.ReadConfig(
                    options=options, try_autocache=False, skip_prefetch=True,
                    num_parallel_calls_for_decode=1,
                    num_parallel_calls_for_interleave_files=1,
                    interleave_cycle_length=2,
                ),
            )
            if world_size > 1:
                dataset = dataset.shard(world_size, rank)
            if worker_info is not None:
                dataset = dataset.shard(worker_info.num_workers, worker_info.id)
            dataset = dataset.shuffle(
                self.shuffle_buffer, seed=epoch_seed, reshuffle_each_iteration=False
            )
            streams.append(iter(tfds.as_numpy(dataset)))
        # Each round visits every remaining suite once in a seeded random order.
        # This balances episodes, not steps: longer episodes still contribute more.
        rng = np.random.default_rng(epoch_seed)
        active = list(range(len(streams)))
        while active:
            for index in rng.permutation(active).tolist():
                try:
                    yield next(streams[index])
                except StopIteration:
                    active.remove(index)
                except Exception as error:
                    # TensorFlow exceptions cannot be reconstructed by PyTorch's
                    # worker error wrapper. Preserve the original message in a
                    # standard exception and never silently drop examples.
                    raise RuntimeError(
                        f"RLDS read failed in {self.dataset_dirs[index]}: "
                        f"{type(error).__name__}: {error}"
                    ) from error

    def _process_image_pair(self, image):
        if isinstance(image, (bytes, np.bytes_)):
            with Image.open(BytesIO(image)) as decoded:
                image = np.asarray(decoded.convert("RGB"))
        return LiberoMixedRLDSDataset._process_image_pair(self, image)
