#!/bin/bash
python run.py \
seed=1 \
logger.project=gabor \
callbacks.save_weights.save_last_epoch_only=True \
datamodule.batch_size=500 \
model.optimizer.name=Adam \
model.optimizer.lr=0.01 \
model.tied_weights=True \
model.activation.name=Sigmoid \
trainer.max_epochs=10 \
model.width=400 \
model.corruption.sigma=0.5