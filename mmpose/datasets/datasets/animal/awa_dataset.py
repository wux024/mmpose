#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project -> File   ：mmpose -> awa_dataset
@Author ：Wu X.
@Email  : wux024@nenu.edu.cn
@Date   ：2024/4/6 10:20
'''
from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class AWADataset(BaseCocoStyleDataset):
    """AnimalAWADataset for animal pose estimation.

        The dataset loads raw features and apply specified transforms
        to return a dict containing the image tensors and other information.

         keypoint indexes::

            "nose",
            "upper_jaw",
            "lower_jaw",
            "mouth_end_right",
            "mouth_end_left",
            "right_eye",
            "right_earbase",
            "right_earend",
            "right_antler_base",
            "right_antler_end",
            "left_eye",
            "left_earbase",
            "left_earend",
            "left_antler_base",
            "left_antler_end",
            "neck_base",
            "neck_end",
            "throat_base",
            "throat_end",
            "back_base",
            "back_end",
            "back_middle",
            "tail_base",
            "tail_end",
            "front_left_thai",
            "front_left_knee",
            "front_left_paw",
            "front_right_thai",
            "front_right_paw",
            "front_right_knee",
            "back_left_knee",
            "back_left_paw",
            "back_left_thai",
            "back_right_thai",
            "back_right_paw",
            "back_right_knee",
            "belly_bottom",
            "body_middle_right",
            "body_middle_left"

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
    METAINFO: dict = dict(from_file='configs/_base_/datasets/awa.py')