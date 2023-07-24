#!/bin/bash
python run.py \
seed=1 \
callbacks=gaussian \
datamodule=gaussian \
logger.project=gabor_synth \
logger.tags='[gaussian]' \
datamodule.n_train=100 \
datamodule.m=100 \
datamodule.k=1 \
model.width=1 \
datamodule.batch_size=100 \
model.optimizer.name=SGD \
model.optimizer.lr=10 \
model.tied_weights=True \
model.activation.name=ReLU \
model.initializer.b_enc=False \
model.initializer.b_dec=False \
trainer.max_epochs=10000 \
model.corruption.sigma=0.0
