#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project -> File   ：mmpose -> ap36k_dataset.py
@Author ：Wu X.
@Email  : wux024@nenu.edu.cn
@Date   ：2024/4/5 12:49
'''
from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class APT36KDataset(BaseCocoStyleDataset):
    """APT-36K dataset for animal pose estimation.

    "AP-10K: A Benchmark for Animal Pose Estimation in the Wild"
    Neurips Dataset Track'2022.
    More details can be found in the `paper
    <https://arxiv.org/abs/2108.12617>`__ .

    APT-36K keypoints::

        0: 'L_Eye',
        1: 'R_Eye',
        2: 'Nose',
        3: 'Neck',
        4: 'root of tail',
        5: 'L_Shoulder',
        6: 'L_Elbow',
        7: 'L_F_Paw',
        8: 'R_Shoulder',
        9: 'R_Elbow',
        10: 'R_F_Paw,
        11: 'L_Hip',
        12: 'L_Knee',
        13: 'L_B_Paw',
        14: 'R_Hip',
        15: 'R_Knee',
        16: 'R_B_Paw'

    Args:
        ann_file (str): Annotation file path. Default: ''.
        bbox_file (str, optional): Detection result file path. If
            ``bbox_file`` is set, detected bboxes loaded from this file will
            be used instead of ground-truth bboxes. This setting is only for
            evaluation, i.e., ignored when ``test_mode`` is ``False``.
            Default: ``None``.
        data_mode (str): Specifies the mode of data samples: ``'topdown'`` or
            ``'bottomup'``. In ``'topdown'`` mode, each data sample contains
            one instance; while in ``'bottomup'`` mode, each data sample
            contains all instances in a image. Default: ``'topdown'``
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Default: ``None``.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Default: ``None``.
        data_prefix (dict, optional): Prefix for training data. Default:
            ``dict(img=None, ann=None)``.
        filter_cfg (dict, optional): Config for filter data. Default: `None`.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Default: ``None`` which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy.
            Default: ``True``.
        pipeline (list, optional): Processing pipeline. Default: [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Default: ``False``.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Default: ``False``.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Default: 1000.
    """

    METAINFO: dict = dict(from_file='configs/_base_/datasets/apt36k.py')