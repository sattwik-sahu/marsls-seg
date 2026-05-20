#!/usr/bin/bash

marsls-train -m \
wandb.group=segmentation \
wandb.mode=online \
data.eval_split="test" \
model.arch=unet,unetplusplus,manet,linknet,deeplabv3plus,fpn \
wandb.artifact_name='seg_ijepa-${model.arch}' \
+run_dir='20260520-003258','20260520-005618','20260520-011841','20260520-014348' \
model.encoder_ckpt_path='${training.ckpt_dir}/${run_dir}/ckpt_ep_500.pt' \
model.encoder_config_path='${training.ckpt_dir}/${run_dir}/config.yaml' \
trainer.apply_aug=false \
training.n_epochs=50 \
training.batch_size=64 \
trainer.n_warmup_epochs=10 \
hydra/launcher=joblib \
hydra.launcher.n_jobs=12
