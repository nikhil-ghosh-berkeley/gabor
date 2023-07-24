#!/bin/bash
python run.py \
seed=1 \
callbacks=default \
datamodule=patches \
logger.project=gabor \
datamodule.batch_size=500 \
model.optimizer.name=SGD \
model.width=400 \
model.optimizer.lr=4 \
model.tied_weights=True \
model.activation.name=ReLU \
trainer.max_epochs=20
