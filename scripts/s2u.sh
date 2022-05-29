#!/bin/bash
python run.py -m \
seed=1 \
callbacks=gabor \
datamodule=gabor \
logger.project=gabor_synth \
datamodule.m=100 \
datamodule.k=10 \
model.width=100 \
datamodule.inc_bound=12 \
datamodule.batch_size=32 \
model.optimizer.name=SGD \
model.optimizer.lr=4 \
model.tied_weights=True \
model.activation.name=ReLU \
trainer.max_epochs=15 \
model.corruption.sigma=0.5
