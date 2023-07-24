#!/bin/bash
python run.py \
seed=1 \
callbacks=gabor \
datamodule=gabor \
callbacks.save_weights.save_last_epoch_only=True \
logger.project=test \
datamodule.batch_size=32 \
datamodule.m=40 \
datamodule.k=3 \
datamodule.inc_bound=3.5 \
model.optimizer.name=SGD \
model.optimizer.lr=2 \
model.tied_weights=True \
model.width=1 \
model.activation.name=ReLU \
trainer.max_epochs=3 \
model.corruption.sigma=0.5
