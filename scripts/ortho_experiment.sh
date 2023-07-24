#!/bin/bash
python run.py \
seed=1 \
callbacks=ortho \
datamodule=ortho \
logger.project=ortho_dict \
datamodule.m=16 \
datamodule.k=1 \
datamodule.gamma=0.5 \
datamodule.patch_width=4 \
datamodule.patch_height=4 \
model.width=1 \
datamodule.batch_size=1 \
datamodule.n_train=10 \
model.optimizer.name=SGD \
model.optimizer.lr=1 \
model.tied_weights=False \
model.activation.name=ReLU \
model.initializer.b_enc=False \
model.initializer.b_dec=False \
trainer.max_epochs=1000 \
model.corruption.sigma=0.0
