#!/bin/bash
python run.py \
seed=1 \
logger.project=test \
datamodule.batch_size=500 \
datamodule.n_train=5000 \
callbacks.save_weights.save_dirs.exp_dir=test \
model.optimizer.name=Adam \
model.optimizer.lr=0.01 \
model.tied_weights=True \
model.activation.name=Sigmoid \
trainer.max_epochs=1 \
model.corruption.sigma=0.5