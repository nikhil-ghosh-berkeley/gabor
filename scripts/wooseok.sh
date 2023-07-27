#!/bin/bash
python run.py -m \
seed=1 \
callbacks=gabor \
datamodule=gabor \
logger.project=slides \
logger.tags='[gabor]' \
datamodule.n_train=600 \
datamodule.n_val=600 \
datamodule.m=15 \
datamodule.k=2 \
datamodule.inc_bound=4 \
datamodule.batch_size=600 \
model.width=25 \
model.optimizer.name=SGD \
model.optimizer.lr=1.0 \
model.tied_weights=True \
model.activation.name=ReLU \
trainer.max_epochs=30000 \
model.corruption.sigma=0
