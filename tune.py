#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os
import shutil
from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from ray import air, train, tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import MedianStoppingRule
from ray.tune.search.ax import AxSearch

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from strhub.data.module import SceneTextDataModule
from strhub.models.base import BaseSystem

log = logging.getLogger(__name__)


class MetricTracker(tune.Stopper):
    """Tracks the trend of the metric. Stops downward/stagnant trials. Assumes metric is being maximized."""

    def __init__(self, metric, max_t, patience: int = 3, window: int = 3) -> None:
        super().__init__()
        self.metric = metric
        self.trial_history = {}
        self.max_t = max_t
        self.training_iteration = 0
        self.eps = 0.01  # sensitivity
        self.patience = patience  # number of consecutive downward/stagnant samples to trigger early stoppage.
        self.kernel = self.gaussian_pdf(np.arange(window) - window // 2, sigma=0.6)
        # Extra samples to keep in order to have better MAs + gradients for the middle p samples.
        self.buffer = 2 * (len(self.kernel) // 2) + 2

    @staticmethod
    def gaussian_pdf(x, sigma=1.0):
        return np.exp(-((x / sigma) ** 2) / 2) / (sigma * np.sqrt(2 * np.pi))

    @staticmethod
    def moving_average(x, k):
        return np.convolve(x, k, 'valid') / k.sum()

    def __call__(self, trial_id, result):
        self.training_iteration = result['training_iteration']
        if np.isnan(result['loss']) or self.training_iteration >= self.max_t:
            try:
                del self.trial_history[trial_id]
            except KeyError:
                pass
            return True
        history = self.trial_history.get(trial_id, [])
        # FIFO queue of metric values.
        history = history[-(self.patience + self.buffer - 1) :] + [result[self.metric]]
        # Only start checking once we have enough data. At least one non-zero sample is required.
        if len(history) == self.patience + self.buffer and sum(history) > 0:
            smooth_grad = np.gradient(self.moving_average(history, self.kernel))[1:-1]  # discard edge values.
            # Check if trend is downward or stagnant
            if (smooth_grad < self.eps).all():
                log.info(f'Stopping trial = {trial_id}, hist = {history}, grad = {smooth_grad}')
                try:
                    del self.trial_history[trial_id]
                except KeyError:
                    pass
                return True
        self.trial_history[trial_id] = history
        return False

    def stop_all(self):
        return False


class TuneReportCheckpointPruneCallback(TuneReportCheckpointCallback):

    def _handle(self, trainer: Trainer, pl_module: LightningModule):
        super()._handle(trainer, pl_module)
        # Prune older checkpoints
        trial_dir = train.get_context().get_trial_dir()
        for old in sorted(Path(trial_dir).glob('checkpoint_epoch=*-step=*'), key=os.path.getmtime)[:-1]:
            log.info(f'Deleting old checkpoint: {old}')
            shutil.rmtree(old)


def trainable(hparams, config):
    with open_dict(config):
        config.model.lr = hparams['lr']
        # config.model.weight_decay = hparams['wd']

    model: BaseSystem = hydra.utils.instantiate(config.model)
    datamodule: SceneTextDataModule = hydra.utils.instantiate(config.data)

    tune_callback = TuneReportCheckpointPruneCallback({
        'loss': 'val_loss',
        'NED': 'val_NED',
        'accuracy': 'val_accuracy',
    })
    if checkpoint := train.get_checkpoint():
        with checkpoint.as_directory() as checkpoint_dir:
            ckpt_path = os.path.join(checkpoint_dir, 'checkpoint')
    else:
        ckpt_path = None
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=TensorBoardLogger(save_dir=train.get_context().get_trial_dir(), name='', version='.'),
        callbacks=[tune_callback],
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


@hydra.main(config_path='configs', config_name='tune', version_base='1.2')
def main(config: DictConfig):
    # Special handling for PARseq
    if config.model.get('perm_mirrored', False):
        assert config.model.perm_num % 2 == 0, 'perm_num should be even if perm_mirrored = True'
    # Modify config
    with open_dict(config):
        # Use mixed-precision training
        if config.trainer.get('gpus', 0):
            config.trainer.precision = 16
        # Resolve absolute path to data.root_dir
        config.data.root_dir = hydra.utils.to_absolute_path(config.data.root_dir)

    hparams = {
        'lr': tune.loguniform(config.tune.lr.min, config.tune.lr.max),
        # 'wd': tune.loguniform(config.tune.wd.min, config.tune.wd.max),
    }

    steps_per_epoch = len(hydra.utils.instantiate(config.data).train_dataloader())
    val_steps = steps_per_epoch * config.trainer.max_epochs / config.trainer.val_check_interval
    max_t = round(0.75 * val_steps)
    warmup_t = round(config.model.warmup_pct * val_steps)
    scheduler = MedianStoppingRule(time_attr='training_iteration', grace_period=warmup_t)

    # Always start by evenly diving the range in log scale.
    lr = hparams['lr']
    start = np.log10(lr.lower)
    stop = np.log10(lr.upper)
    num = math.ceil(stop - start) + 1
    initial_points = [{'lr': np.clip(x, lr.lower, lr.upper).item()} for x in reversed(np.logspace(start, stop, num))]
    search_alg = AxSearch(points_to_evaluate=initial_points)

    reporter = CLIReporter(parameter_columns=['lr'], metric_columns=['loss', 'accuracy', 'training_iteration'])

    out_dir = Path(HydraConfig.get().runtime.output_dir if config.tune.resume_dir is None else config.tune.resume_dir)

    resources_per_trial = {
        'cpu': 1,
        'gpu': config.tune.gpus_per_trial,
    }

    wrapped_trainable = tune.with_parameters(tune.with_resources(trainable, resources_per_trial), config=config)
    if config.tune.resume_dir is None:
        tuner = tune.Tuner(
            wrapped_trainable,
            param_space=hparams,
            tune_config=tune.TuneConfig(
                mode='max',
                metric='NED',
                search_alg=search_alg,
                scheduler=scheduler,
                num_samples=config.tune.num_samples,
            ),
            run_config=air.RunConfig(
                name=out_dir.name,
                stop=MetricTracker('NED', max_t),
                progress_reporter=reporter,
                storage_path=str(out_dir.parent.absolute()),
            ),
        )
    else:
        tuner = tune.Tuner.restore(config.tune.resume_dir, wrapped_trainable)
    results = tuner.fit()
    best_result = results.get_best_result()

    print('Best hyperparameters found were:', best_result.config)
    print('with result:\n', best_result)


if __name__ == '__main__':
    main()
