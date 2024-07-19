#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project -> File   ：mmpose -> standfordextra_dataset
@Author ：Wu X.
@Email  : wux024@nenu.edu.cn
@Date   ：2024/4/6 13:20
'''
from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class StanfordExtraDataset(BaseCocoStyleDataset):
    """StanfordExtraDataset for animal pose estimation.

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    Dog keypoint indexes::
        "FrontLeftLeg0",
        "FrontLeftLeg1",
        "FrontLeftLeg2",
        "FrontRightLeg0",
        "FrontRightLeg1",
        "FrontRightLeg2",
        "BackLeftLeg0",
        "BackLeftLeg1",
        "BackLeftLeg2",
        "BackRightLeg0",
        "BackRightLeg1",
        "BackRightLeg2",
        "Tail0",
        "Tail1",
        "LeftEar0",
        "RightEar0",
        "Spout0",
        "Spout1",
        "LeftEar1",
        "RightEar1",

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

    METAINFO: dict = dict(from_file='configs/_base_/datasets/stanfordextra.py')