from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from fish_diffusion.datasets.utils import get_datasets_from_subfolder
from fish_diffusion.datasets.naive import NaiveSVCDataset
from fish_diffusion.utils.pitch import pitch_to_log

_base_ = [
    "./_base_/archs/diff_svc_v2_lstm2.py",
    "./_base_/trainers/base.py",
    "./_base_/schedulers/warmup_cosine_finetune.py",
    "./_base_/datasets/naive_svc.py",
]

speaker_mapping = {
    "neuro": 0,
    "tsukuyomi": 1,
    "basic": 2,
    "m4sop": 3,
    "m4alt": 4,
    "man": 5,
}

dataset = dict(
    train=dict(
        _delete_=True,  # Delete the default train dataset
        type="ConcatDataset",
        datasets=get_datasets_from_subfolder(
            "NaiveSVCDataset",
            "dataset/train",
            speaker_mapping,
        ),
        # + mixin_datasets,
        collate_fn=NaiveSVCDataset.collate_fn,
    ),
    valid=dict(
        _delete_=True,  # Delete the default train dataset
        type="ConcatDataset",
        datasets=get_datasets_from_subfolder(
            "NaiveSVCDataset",
            "dataset/valid",
            speaker_mapping,
        ),
        collate_fn=NaiveSVCDataset.collate_fn,
    ),
)

model = dict(
    text_encoder=dict(
        type="NaiveProjectionEncoder",
        input_size=256,
        output_size=256,
    ),
    speaker_encoder=dict(
        input_size=len(speaker_mapping),
    ),
    pitch_encoder=dict(
        preprocessing=pitch_to_log,
    ),
    pitch_shift_encoder=dict(
        type="NaiveProjectionEncoder",
        input_size=1,
        output_size=256,
        use_embedding=False,
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
        f0_max=2000.0,
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

# The following trainer val and save checkpoints every 1000 steps
trainer = dict(
    val_check_interval=1000,
    callbacks=[
        ModelCheckpoint(
            filename="{epoch}-{step}-{valid_loss:.2f}",
            every_n_train_steps=1000,
            save_top_k=-1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ],
)
