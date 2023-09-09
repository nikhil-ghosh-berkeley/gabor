#!/bin/bash
# synthetic Gabor dictionary. 
# Interestingly, disabling encoder bias (model.initializer.b_enc=False) gives perfect recovery after 1 epoch. 
# keeping bias gets close but not perfect.
python run.py \
seed=1 \
callbacks=gabor \
datamodule=gabor \
logger.project=gabor_synth \
datamodule.batch_size=32 \
datamodule.m=40 \
datamodule.k=3 \
datamodule.inc_bound=3.5 \
model.optimizer.name=SGD \
model.optimizer.lr=2 \
model.tied_weights=True \
model.width=40 \
model.activation.name=ReLU \
trainer.max_epochs=1 \
model.initializer.b_enc=False \
model.corruption.sigma=0.5
