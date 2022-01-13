#!/bin/bash
python run.py \
decay_time=1 \
datamodule.batch_size=250 \
model.optimizer_partial._args_=['${get_method:torch.optim.Adam}'] \
model.optimizer_partial.lr=0.01 \
trainer.max_epochs=10
