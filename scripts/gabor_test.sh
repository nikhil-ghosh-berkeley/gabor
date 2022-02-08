#!/bin/bash
python run.py \
seed=1 \
callbacks=gabor \
datamodule=gabor \
logger.project=test \
datamodule.batch_size=250 \
datamodule.m=40 \
datamodule.k=10 \
datamodule.inc_bound=3.5 \
model.optimizer.name=Adam \
model.optimizer.lr=0.01 \
model.tied_weights=True \
model.width=50 \
model.activation.name=Sigmoid \
trainer.max_epochs=1 \
model.corruption.sigma=0.5
