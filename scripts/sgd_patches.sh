#!/bin/bash
python run.py \
seed=1 \
datamodule.batch_size=32 \
model.optimizer.name=SGD \
model.optimizer.lr=2 \
model.tied_weights=True \
model.activation.name=Sigmoid \
trainer.max_epochs=128 \
model.corruption.sigma=0.5
