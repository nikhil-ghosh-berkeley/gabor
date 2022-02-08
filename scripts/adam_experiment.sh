#!/bin/bash
python run.py -m \
seed=1 \
logger.project=gabor \
datamodule.batch_size=500 \
model.optimizer.name=Adam \
model.optimizer.lr=0.0025 \
model.tied_weights=True \
model.activation.name=Sigmoid \
trainer.max_epochs=100 \
model.width=400 \
model.corruption.sigma=0.5