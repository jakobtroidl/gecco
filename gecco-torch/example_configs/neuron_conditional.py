import os

import torch
import lightning.pytorch as pl

from gecco_torch.diffusion import EDMPrecond, Diffusion, IdleConditioner
from gecco_torch.reparam import GaussianReparam
from gecco_torch.diffusion import Diffusion, LogUniformSchedule, EDMLoss
from gecco_torch.models.set_transformer import SetTransformer
from gecco_torch.models.partial_lift import LinearLiftCond
from gecco_torch.models.part_enc import PointEncoder
from gecco_torch.models.activation import GaussianActivation
from gecco_torch.data.neurons import NeuronDataModule
from gecco_torch.ema import EMACallback
from gecco_torch.vis import PCVisCallback

dataset_path = (
    "/nrs/turaga/jakob/autoproof_data/flywire_cave_post/cache_t5"
)
data = NeuronDataModule(
    dataset_path,
    epoch_size=5_000,
    batch_size=128,
    num_workers=16,
)

reparam = GaussianReparam(
    mean=torch.tensor([0.6815091371536255, 0.2573479115962982, 1.0269641876220703]),
    sigma=torch.tensor([0.027468865737318993, 0.04374050721526146, 0.1090131476521492]),
)

feature_dim = 3 * 128
network = LinearLiftCond(
    inner=SetTransformer(
        n_layers=6,
        num_inducers=64,
        feature_dim=feature_dim,
        t_embed_dim=1,
        num_heads=8,
        activation=GaussianActivation,
    ),
    feature_dim=feature_dim,
)

model = Diffusion(
    backbone=EDMPrecond(
        model=network,
    ),
    conditioner=PointEncoder(),
    reparam=reparam,
    loss=EDMLoss(
        schedule=LogUniformSchedule(
            max=219.14358520507812,  # from find_hyperparameters.ipynb
        ),
    ),
)


def trainer():
    return pl.Trainer(
        default_root_dir=os.path.split(__file__)[0],
        callbacks=[
            EMACallback(decay=0.99),
            pl.callbacks.ModelCheckpoint(),
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                filename="{epoch}-{val_loss:.3f}",
                save_top_k=3,
                mode="min",
            ),
            # PCVisCallback(n=8, n_steps=128, point_size=0.01),
        ],
        max_epochs=50,
        # precision=32,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="value",
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    model = torch.compile(model)
    trainer().fit(model, data)
