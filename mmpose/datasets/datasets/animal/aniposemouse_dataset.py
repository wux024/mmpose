#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project -> File   ：mmpose -> aniposemouse
@Author ：Wu X.
@Email  : wux024@nenu.edu.cn
@Date   ：2024/4/6 13:54
'''
from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class AniposeMouseDataset(BaseCocoStyleDataset):

    METAINFO: dict = dict(from_file='configs/_base_/datasets/aniposemouse.py')