#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project -> File   ：mmpose -> mouse_dataset
@Author ：Wu X.
@Email  : wux024@nenu.edu.cn
@Date   ：2024/6/25 12:40
'''
from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class TopViewMouseDataset(BaseCocoStyleDataset):

    METAINFO: dict = dict(from_file='configs/_base_/datasets/topviewmouse.py')