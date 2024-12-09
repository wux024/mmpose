#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project -> File   ：mmpose -> mouse
@Author ：Wu X.
@Email  : wux024@nenu.edu.cn
@Date   ：2024/4/6 14:15
'''
dataset_info = dict(
    dataset_name='Mouse',
    paper_info=dict(
        author='Mathis, Alexander and Biasi, Thomas and '
        'Schneider, Steffen and '
        'Yuksekgonul, Mert and Rogers, Byron and '
        'Bethge, Matthias and '
        'Mathis, Mackenzie W',
        title='Pretraining boosts out-of-domain robustness '
        'for pose estimation',
        container='Proceedings of the IEEE/CVF Winter Conference on '
        'Applications of Computer Vision',
        year='2021',
        homepage='http://www.mackenziemathislab.org/horse10',
    ),
    keypoint_info={
        0:
            dict(
                name='snout', id=0, color=[0, 255, 0], type='upper', swap=''),
        1:
            dict(
                name='leftear',
                id=1,
                color=[255, 128, 0],
                type='upper',
                swap='rightear'),
        2:
            dict(name='rightear', id=2, color=[51, 153, 255], type='upper', swap='leftear'),
        3:
            dict(name='shoulder', id=3, color=[51, 153, 255], type='upper', swap=''),
        4:
            dict(
                name='spine1',
                id=4,
                color=[51, 153, 255],
                type='lower',
                swap=''),
        5:
            dict(
                name='spine2',
                id=5,
                color=[51, 153, 255],
                type='upper',
                swap=''),
        6:
            dict(
                name='spine3',
                id=6,
                color=[51, 153, 255],
                type='upper',
                swap=''),
        7:
            dict(
                name='spine4',
                id=7,
                color=[0, 255, 0],
                type='upper',
                swap=''),
        8:
            dict(
                name='tailbase',
                id=8,
                color=[0, 255, 0],
                type='upper',
                swap=''),
        9:
            dict(
                name='tail1',
                id=9,
                color=[255, 128, 0],
                type='upper',
                swap=''),
        10:
            dict(
                name='tail2',
                id=10,
                color=[0, 255, 0],
                type='lower',
                swap=''),
        11:
            dict(
                name='tailend',
                id=11,
                color=[255, 128, 0],
                type='lower',
                swap=''),
    },
    skeleton_info={
        0: dict(link=('snout', 'leftear'), id=0, color=[0, 0, 255]),
        1: dict(link=('snout', 'rightear'), id=1, color=[0, 0, 255]),
        2: dict(link=('shoulder', 'spine1'), id=2, color=[0, 0, 255]),
        3: dict(link=('spine1', 'spine2'), id=3, color=[0, 255, 0]),
        4: dict(link=('spine2', 'spine3'), id=4, color=[0, 255, 0]),
        5: dict(link=('spine3', 'spine4'), id=6, color=[0, 255, 255]),
        6: dict(link=('spine4', 'tailbase'), id=7, color=[0, 255, 255]),
        7: dict(link=('tailbase', 'tail1'), id=9, color=[6, 156, 250]),
        8: dict(link=('tail1', 'tail2'), id=10, color=[6, 156, 250]),
        9: dict(link=('tail2', 'tailend'), id=11, color=[0, 255, 255]),
    },
    joint_weights=[1.] * 12,
    sigmas=[1.0 / 12.0] * 12)