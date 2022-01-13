#!/bin/bash
python run.py \
decay_time=1 \
datamodule.batch_size=32 \
model.optimizer_partial._args_=['${get_method:torch.optim.SGD}'] \
model.optimizer_partial.lr=2.0 \
trainer.max_epochs=128