#!/bin/bash
python run.py \
seed=1 \
callbacks=ortho \
datamodule=ortho \
logger.project=ortho_dict \
datamodule.m=40 \
datamodule.k=1 \
model.width=1 \
datamodule.batch_size=100 \
datamodule.n_train=50000 \
model.optimizer.name=SGD \
model.optimizer.lr=1 \
model.tied_weights=True \
model.activation.name=ReLU \
trainer.max_epochs=100 \
trainer.val_check_interval=1.0 \
model.corruption.sigma=0.1
