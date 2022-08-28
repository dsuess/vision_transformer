from typing import Any, Mapping, Optional

import numpy as np
import wandb.plot
from clu.metric_writers import LoggingWriter, MultiWriter, SummaryWriter
from clu.metric_writers.interface import MetricWriter, Scalar
from wandb.sdk.wandb_run import Run

import wandb


class WandbWriter(MetricWriter):
    def __init__(self, wandb_project: str):
        super().__init__()
        self.run: Run = wandb.init(project=wandb_project)

    def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
        self.run.log(scalars, step=step)

    @staticmethod
    def _make_histogram_table(name: str, data: np.ndarray) -> Any:
        if data.ndim != 1:
            raise ValueError(f"Can't deal with data of shape {data.shape}")
        table = wandb.Table(data=data[:, None], columns=[name])
        return wandb.plot.histogram(table, name, title=f"Histogram of {name}")

    def write_histograms(
        self,
        step: int,
        arrays: Mapping[str, np.ndarray],
        num_buckets: Optional[Mapping[str, int]] = None,
    ):
        if num_buckets is not None:
            raise NotImplementedError("Custom buckets not implemented")

        self.run.log(
            {
                f"histograms/{name}": self._make_histogram_table(name, data)
                for name, data in arrays.items()
            },
            step=step,
        )

    def write_images(self, step: int, images: Mapping[str, np.ndarray]):
        raise NotImplementedError()

    def write_texts(self, step: int, texts: Mapping[str, str]):
        raise NotImplementedError()

    def write_hparams(self, hparams: Mapping[str, Any]):
        self.run.config.update(hparams)

    def flush(self):
        super().flush()
        self.run.save()

    def close(self):
        super().close()
        self.run.finish()


def create_default_writer(logdir: str, *, wandb_project: str = None) -> MetricWriter:
    writers = [LoggingWriter(), SummaryWriter(logdir)]

    if wandb_project is not None:
        writers += [WandbWriter(wandb_project)]
    return MultiWriter(writers)
