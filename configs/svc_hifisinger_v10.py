from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from fish_diffusion.datasets.hifisinger import HiFiSVCDataset

_base_ = [
    "./_base_/archs/hifi_svc.py",
    "./_base_/trainers/base.py",
    "./_base_/schedulers/exponential.py",
    "./_base_/datasets/hifi_svc.py",
]

speaker_mapping = {
    "neuro": 0,
    "tsukuyomi": 1,
    "basic": 2,
    "m4sop": 3,
    "m4sop2": 4,
    "m4alt": 5,
}

dataset = dict(
    train=dict(
        _delete_=True,  # Delete the default train dataset
        type="ConcatDataset",
        datasets=[
            dict(
                type="HiFiSVCDataset",
                path="dataset/train/neuro",
                speaker_id=speaker_mapping["neuro"],
                segment_size=16384
            ),
            dict(
                type="HiFiSVCDataset",
                path="dataset/train/tsukuyomi",
                speaker_id=speaker_mapping["tsukuyomi"],
                segment_size=16384
            ),
            dict(
                type="HiFiSVCDataset",
                path="dataset/train/basic",
                speaker_id=speaker_mapping["basic"],
                segment_size=16384
            ),
            dict(
                type="HiFiSVCDataset",
                path="dataset/train/m4sop",
                speaker_id=speaker_mapping["m4sop"],
                segment_size=16384
            ),
            dict(
                type="HiFiSVCDataset",
                path="dataset/train/m4sop2",
                speaker_id=speaker_mapping["m4sop2"],
                segment_size=16384
            ),
            dict(
                type="HiFiSVCDataset",
                path="dataset/train/m4alt",
                speaker_id=speaker_mapping["m4alt"],
                segment_size=16384
            ),
        ],
        collate_fn=HiFiSVCDataset.collate_fn,
    ),
    valid=dict(
        _delete_=True,  # Delete the default train dataset
        type="ConcatDataset",
        datasets=[
            dict(
                type="HiFiSVCDataset",
                path="dataset/valid/neuro",
                speaker_id=speaker_mapping["neuro"],
                segment_size=-1
            ),
            dict(
                type="HiFiSVCDataset",
                path="dataset/valid/tsukuyomi",
                speaker_id=speaker_mapping["tsukuyomi"],
                segment_size=-1
            ),
            dict(
                type="HiFiSVCDataset",
                path="dataset/valid/basic",
                speaker_id=speaker_mapping["basic"],
                segment_size=-1
            ),
            dict(
                type="HiFiSVCDataset",
                path="dataset/valid/m4sop",
                speaker_id=speaker_mapping["m4sop"],
                segment_size=-1
            ),
            dict(
                type="HiFiSVCDataset",
                path="dataset/valid/m4sop2",
                speaker_id=speaker_mapping["m4sop2"],
                segment_size=-1
            ),
            dict(
                type="HiFiSVCDataset",
                path="dataset/valid/m4alt",
                speaker_id=speaker_mapping["m4alt"],
                segment_size=-1
            ),
        ],
        collate_fn=HiFiSVCDataset.collate_fn,
    ),
)

model = dict(
    type="HiFiSVC",
    speaker_encoder=dict(
        input_size=len(speaker_mapping),
    ),
)

preprocessing = dict(
    text_features_extractor=dict(
        type="ContentVec",
    ),
    pitch_extractor=dict(
        type="CrepePitchExtractor",
        keep_zeros=False,
        f0_min=40.0,
        f0_max=1600.0,
    ),
    energy_extractor=dict(
        type="RMSEnergyExtractor",
    ),
    augmentations=[
        dict(
            type="FixedPitchShifting",
            key_shifts=[-5.0, 5.0],
            probability=1.5,
        ),
    ],
)

trainer = dict(
    # Disable gradient clipping, which is not supported by custom optimization
    gradient_clip_val=None,
    val_check_interval=1000,
    check_val_every_n_epoch=None,
    callbacks=[
        ModelCheckpoint(
            filename="{epoch}-{step}-{valid_loss:.2f}",
            every_n_train_steps=1000,
            save_top_k=-1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ],
)
