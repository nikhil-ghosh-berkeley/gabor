#!/bin/bash
python run.py -m \
seed=1 \
callbacks=gabor \
datamodule=gabor \
logger.project=gabor_synth \
logger.tags='[s2o]' \
datamodule.m=200 \
datamodule.k=3,10 \
model.width=400 \
datamodule.inc_bound=12 \
datamodule.batch_size=250 \
model.optimizer.name=Adam \
model.optimizer.lr=0.01 \
model.tied_weights=True \
model.activation.name=Sigmoid \
trainer.max_epochs=15 \
model.corruption.sigma=0.5
