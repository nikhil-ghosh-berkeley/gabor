#!/bin/bash
python run.py -m \
seed=1 \
callbacks=gabor \
datamodule=gabor \
logger.project=gabor_synth \
datamodule.m=1000 \
datamodule.k=10 \
model.width=200,400 \
datamodule.inc_bound=12 \
datamodule.batch_size=250 \
model.optimizer.name=Adam \
model.optimizer.lr=0.005 \
model.tied_weights=True \
model.activation.name=Sigmoid \
trainer.max_epochs=50 \
model.corruption.sigma=0.5
