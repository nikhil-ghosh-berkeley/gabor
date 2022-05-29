#!/bin/bash
python run.py -m \
seed=1 \
callbacks=gaussian \
datamodule=gaussian \
logger.project=gabor_synth \
logger.tags='[gaussian]' \
datamodule.m=210 \
datamodule.k=10 \
model.width=210 \
datamodule.batch_size=250 \
model.optimizer.name=Adam \
model.optimizer.lr=0.01 \
model.tied_weights=True \
model.activation.name=Sigmoid \
trainer.max_epochs=10 \
model.corruption.sigma=0,0.5
