#!/bin/bash
python run.py -m \
seed=1 \
callbacks=gabor \
datamodule=gabor \
logger.project=slides_new \
logger.tags='[gabor]' \
datamodule.n_train=200 \
datamodule.n_val=200 \
datamodule.m=10 \
datamodule.k=2 \
datamodule.inc_bound=10 \
datamodule.batch_size=25 \
model.width=10 \
model.optimizer.name=SGD \
model.optimizer.lr=2.0 \
model.tied_weights=True \
model.activation.name=ReLU \
trainer.max_epochs=2000 \
model.corruption.sigma=0
