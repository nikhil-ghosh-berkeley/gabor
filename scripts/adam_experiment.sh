#!/bin/bash
python run.py \
seed=1 \
callbacks.save_weights.save_dir='${work_dir}/good_filters' \
datamodule.batch_size=500 \
model.optimizer.name=Adam \
model.optimizer.lr=0.01 \
model.tied_weights=True \
model.activation.name=Sigmoid \
trainer.max_epochs=25 \
model.corruption.sigma=0.5