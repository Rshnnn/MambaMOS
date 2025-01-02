_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 8  # bs: total bs in all gpus  # 1 batch/gpu
lr_per_sample = 0.00008
# gride_size = 0.09
gride_size = 1
event_size = [640 , 360]

gather_num = 3
epoch = 200
eval_epoch = 50


seed = 1024  # train process will init a random seed and record
num_worker = 8
empty_cache = False
enable_amp = False

# dataset settings
data_root = "/media/sdb2/grs/data/EVT3/crop/scene1"
# data_root = "data/sythetic/SYTHETIC_DATA_DIR"
# data_root = "/your/path/to/semantickitti"
num_classes = 4
ignore_index = 0
names = ["unlabeled", "static", "movable", "moving"]
dataset_type = "SemanticKITTIMultiScansDataset"

shuffle_orders = False
scan_modulation = False

# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=8,
    backbone=dict(
        type="MambaMOS",
        in_channels=5,  # x, y, z, i, t
        gather_num=gather_num,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        # stride=(2, 2, 2, 2),
        # enc_depths=(2, 2, 2, 6, 2),
        # enc_channels=(32, 64, 128, 256, 512),
        # dec_depths=(2, 2, 2, 2),
        # dec_channels=(64, 64, 128, 256),
        stride=(2),
        enc_depths=(2, 2),
        enc_channels=(8, 16),
        dec_depths=(2),
        dec_channels=(8),
        mlp_ratio=4,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=shuffle_orders,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        # dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
        # dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=ignore_index),
        dict(type="ChamferDistance", loss_weight=1.0),
        dict(type="IntensityLoss",loss_weight=0.5),
        # dict(type="chamfer_loss_with_intensity", loss_weight=1.0),
        # dict(type="hausdorff_loss_with_intensity", loss_weight=1.0),
    ],
)

# scheduler settings
optimizer = dict(type="AdamW", lr=lr_per_sample * batch_size, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[lr_per_sample * batch_size, lr_per_sample * batch_size * 0.1],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=lr_per_sample * batch_size * 0.1)]

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        # data_width = 460,
        # data_height = 352,
        # data_width = 1280,
        # data_height = 720,
        data_width = event_size[0],
        data_height = event_size[1],
        gather_num=gather_num,
        scan_modulation=scan_modulation,
        transform=[
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            # dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSample",
                grid_size=gride_size,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment", "tn"),
                return_grid_coord=True,
            ),
            # dict(type="Normalize", resolution=[640, 360]),
            dict(type="Normalize", resolution=[event_size[0], event_size[1]]),
            
            # dict(type="Normalize", resolution=[460, 352]),
            # dict(type="PointClip", point _cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
            dict(type="SphereCrop", sample_rate=0.8, mode="random"),

            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "tn"),
                feat_keys=("coord", "strength", "tn"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        gather_num=gather_num,
        scan_modulation=scan_modulation,
        transform=[
            dict(
                type="GridSample",
                grid_size=gride_size,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment", "tn"),
                return_grid_coord=True,
            ),
            # dict(type="Normalize", resolution=[640, 360]),
            dict(type="Normalize", resolution=[event_size[0], event_size[1]]),
            # dict(type="Normalize", resolution=[460, 352]),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "tn"),
                feat_keys=("coord", "strength", "tn"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),

    test=dict(
        type=dataset_type,
        split="test",
        event_size = [640 , 360],
        data_root=data_root,
        gather_num=gather_num,
        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment", "tn": "origin_tn"}),
            dict(
                type='GridSample',
                grid_size=1.5,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'strength', 'segment', 'tn'),
                return_inverse=True,
                return_grid_coord=True),
            # dict(type="Normalize", resolution=[460, 352]),
            dict(type="Normalize", resolution=[event_size[0], event_size[1]]),
        ],
        test_mode=True,
        test_cfg=dict(
            scale2kitti=None,
            voxelize=None,
            crop=None,
            post_transform=[
            dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "tn"),
                    feat_keys=("coord", "strength", "tn"),
                ),
            ],
            aug_transform=[
                [
                    dict(type="Add")
                ],
            ],
        ),
        ignore_index=ignore_index,
    ),

    # test=dict(
    #     type=dataset_type,
    #     split="test",
    #     data_root=data_root,
    #     gather_num=gather_num,
    #     transform=[
    #         dict(type="Copy", keys_dict={"segment": "origin_segment", "tn": "origin_tn"}),
    #         dict(
    #             type="GridSample",
    #             grid_size=gride_size / 2,
    #             hash_type="fnv",
    #             mode="train",
    #             keys=("coord", "strength", "segment", "tn"),
    #             return_inverse=True,
    #             ########
    #         ),
    #         # dict(type="Normalize", resolution=[640, 360]),
    #         dict(type="Normalize", resolution=[1280, 720]),
    #         # dict(type="Normalize", resolution=[460, 352]),
    #     ],
    #     test_mode=True,
    #     test_cfg=dict(
    #         scale2kitti=None,
    #         voxelize=None,
    #         crop=None,
    #         post_transform=[
    #             dict(type="ToTensor"),
    #             dict(
    #                 type="Collect",
    #                 keys=("coord", "grid_coord", "index", "tn"),
    #                 feat_keys=("coord", "strength", "tn"),
    #             ),
    #         ],
    #         aug_transform=[
    #             [
    #                 dict(type="Add")
    #             ],
    #         ],
    #     ),
    #     ignore_index=ignore_index,
    # ),
)
