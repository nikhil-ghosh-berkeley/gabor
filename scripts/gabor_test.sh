#!/bin/bash
python run.py \
seed=1 \
callbacks=gabor \
datamodule=gabor \
logger.project=test \
datamodule.batch_size=250 \
datamodule.n_train=500000 \
model.optimizer.name=Adam \
model.optimizer.lr=0.01 \
model.tied_weights=True \
model.activation.name=Sigmoid \
trainer.max_epochs=10 \
model.corruption.sigma=0.5
