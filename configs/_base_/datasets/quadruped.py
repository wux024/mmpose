#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project -> File   ：mmpose -> quadruped.py
@Author ：Wu X.
@Email  : wux024@nenu.edu.cn
@Date   ：2024/6/25 12:00
'''

dataset_info = dict(
    dataset_name="Quadruped80K",
    paper_info=dict(
        author=' Shaokai Ye and Anastasiia Filippova and Jessy Lauer and '
                'Steffen Schneider and Maxime Vidal and Tian Qiu and '
                'Alexander Mathis and Mackenzie Weygandt Mathis',
        title='SuperAnimal pretrained pose estimation models for behavioral analysis',
        container='Nature Communications',
        year='2024',
        homepage='https://zenodo.org/records/10619173'
    ),
    keypoint_info = {
            0: {
                "name": "nose",
                "id": 0,
                "color": [
                    255.0,
                    0.0,
                    0.0
                ],
                "type": "upper",
                "swap": ""
            },
            1: {
                "name": "upper_jaw",
                "id": 1,
                "color": [
                    255.0,
                    39.63408568671726,
                    0.0
                ],
                "type": "upper",
                "swap": ""
            },
            2: {
                "name": "lower_jaw",
                "id": 2,
                "color": [
                    255.0,
                    79.26817137343453,
                    0.0
                ],
                "type": "upper",
                "swap": ""
            },
            3: {
                "name": "mouth_end_right",
                "id": 3,
                "color": [
                    255.0,
                    118.9022570601518,
                    0.0
                ],
                "type": "upper",
                "swap": "mouth_end_left"
            },
            4: {
                "name": "mouth_end_left",
                "id": 4,
                "color": [
                    255.0,
                    158.53634274686905,
                    0.0
                ],
                "type": "upper",
                "swap": "mouth_end_right"
            },
            5: {
                "name": "right_eye",
                "id": 5,
                "color": [
                    255.0,
                    198.17042843358632,
                    0.0
                ],
                "type": "upper",
                "swap": "left_eye"
            },
            6: {
                "name": "right_earbase",
                "id": 6,
                "color": [
                    255.0,
                    237.8045141203036,
                    0.0
                ],
                "type": "upper",
                "swap": "left_earbase"
            },
            7: {
                "name": "right_earend",
                "id": 7,
                "color": [
                    232.56140019297916,
                    255.0,
                    0.0
                ],
                "type": "upper",
                "swap": "left_earend"
            },
            8: {
                "name": "right_antler_base",
                "id": 8,
                "color": [
                    192.92731450626187,
                    255.0,
                    0.0
                ],
                "type": "upper",
                "swap": "left_antler_base"
            },
            9: {
                "name": "right_antler_end",
                "id": 9,
                "color": [
                    153.2932288195446,
                    255.0,
                    0.0
                ],
                "type": "upper",
                "swap": "left_antler_end"
            },
            10: {
                "name": "left_eye",
                "id": 10,
                "color": [
                    113.65914313282731,
                    255.0,
                    0.0
                ],
                "type": "upper",
                "swap": "right_eye"
            },
            11: {
                "name": "left_earbase",
                "id": 11,
                "color": [
                    74.02505744611004,
                    255.0,
                    0.0
                ],
                "type": "upper",
                "swap": "right_earbase"
            },
            12: {
                "name": "left_earend",
                "id": 12,
                "color": [
                    34.390971759392784,
                    255.0,
                    0.0
                ],
                "type": "upper",
                "swap": "right_earend"
            },
            13: {
                "name": "left_antler_base",
                "id": 13,
                "color": [
                    3.5647953575585385,
                    255.0,
                    8.807909284882923
                ],
                "type": "upper",
                "swap": "right_antler_base"
            },
            14: {
                "name": "left_antler_end",
                "id": 14,
                "color": [
                    0.0,
                    255.0,
                    44.87701729490043
                ],
                "type": "upper",
                "swap": "right_antler_end"
            },
            15: {
                "name": "neck_base",
                "id": 15,
                "color": [
                    0.0,
                    255.0,
                    84.51085328820125
                ],
                "type": "upper",
                "swap": ""
            },
            16: {
                "name": "neck_end",
                "id": 16,
                "color": [
                    0.0,
                    255.0,
                    124.14468928150207
                ],
                "type": "upper",
                "swap": ""
            },
            17: {
                "name": "throat_base",
                "id": 17,
                "color": [
                    0.0,
                    255.0,
                    163.77852527480275
                ],
                "type": "upper",
                "swap": ""
            },
            18: {
                "name": "throat_end",
                "id": 18,
                "color": [
                    0.0,
                    255.0,
                    203.4123612681037
                ],
                "type": "upper",
                "swap": ""
            },
            19: {
                "name": "back_base",
                "id": 19,
                "color": [
                    0.0,
                    255.0,
                    243.04619726140453
                ],
                "type": "upper",
                "swap": ""
            },
            20: {
                "name": "back_end",
                "id": 20,
                "color": [
                    0,
                    220,
                    255
                ],
                "type": "upper",
                "swap": ""
            },
            21: {
                "name": "back_middle",
                "id": 21,
                "color": [
                    0,
                    255,
                    255
                ],
                "type": "upper",
                "swap": ""
            },
            22: {
                "name": "tail_base",
                "id": 22,
                "color": [
                    0,
                    165,
                    255
                ],
                "type": "upper",
                "swap": ""
            },
            23: {
                "name": "tail_end",
                "id": 23,
                "color": [
                    0,
                    150,
                    255
                ],
                "type": "upper",
                "swap": ""
            },
            24: {
                "name": "front_left_thai",
                "id": 24,
                "color": [
                    0.0,
                    68.78344961404169,
                    255.0
                ],
                "type": "upper",
                "swap": "front_right_thai"
            },
            25: {
                "name": "front_left_knee",
                "id": 25,
                "color": [
                    0.0,
                    29.14936392732455,
                    255.0
                ],
                "type": "upper",
                "swap": "front_right_knee"
            },
            26: {
                "name": "front_left_paw",
                "id": 26,
                "color": [
                    10.484721759392611,
                    0.0,
                    255.0
                ],
                "type": "upper",
                "swap": "front_right_paw"
            },
            27: {
                "name": "front_right_thai",
                "id": 27,
                "color": [
                    50.11880744611004,
                    0.0,
                    255.0
                ],
                "type": "upper",
                "swap": "front_left_thai"
            },
            28: {
                "name": "front_right_knee",
                "id": 28,
                "color": [
                    89.75289313282732,
                    0.0,
                    255.0
                ],
                "type": "upper",
                "swap": "front_left_knee"
            },
            29: {
                "name": "front_right_paw",
                "id": 29,
                "color": [
                    129.38697881954448,
                    0.0,
                    255.0
                ],
                "type": "upper",
                "swap": "front_left_paw"
            },
            30: {
                "name": "back_left_paw",
                "id": 30,
                "color": [
                    169.02106450626192,
                    0.0,
                    255.0
                ],
                "type": "upper",
                "swap": "back_right_paw"
            },
            31: {
                "name": "back_left_thai",
                "id": 31,
                "color": [
                    169.02106450626192,
                    0.0,
                    255.0
                ],
                "type": "upper",
                "swap": "back_right_thai"
            },
            32: {
                "name": "back_right_thai",
                "id": 32,
                "color": [
                    255.0,
                    0.0,
                    142.80850706015173
                ],
                "type": "upper",
                "swap": "back_left_thai"
            },
            33: {
                "name": "back_left_knee",
                "id": 33,
                "color": [
                    169.02106450626192,
                    0.0,
                    255.0
                ],
                "type": "upper",
                "swap": "back_right_knee"
            },
            34: {
                "name": "back_right_knee",
                "id": 34,
                "color": [
                    255.0,
                    0.0,
                    142.80850706015173
                ],
                "type": "upper",
                "swap": "back_left_knee"
            },
            35: {
                "name": "back_right_paw",
                "id": 35,
                "color": [
                    255.0,
                    0.0,
                    142.80850706015173
                ],
                "type": "upper",
                "swap": "back_left_paw"
            },
            36: {
                "name": "belly_bottom",
                "id": 36,
                "color": [
                    255.0,
                    0.0,
                    103.17442137343447
                ],
                "type": "upper",
                "swap": ""
            },
            37: {
                "name": "body_middle_right",
                "id": 37,
                "color": [
                    255.0,
                    0.0,
                    63.54033568671722
                ],
                "type": "upper",
                "swap": "body_middle_left"
            },
            38: {
                "name": "body_middle_left",
                "id": 38,
                "color": [
                    255.0,
                    0.0,
                    23.90625
                ],
                "type": "upper",
                "swap": "body_middle_right"
            }
        },
        joint_weights=[1.0]*39,
        sigmas=[
            0.026,
            0.067,
            0.067,
            0.067,
            0.067,
            0.025,
            0.067,
            0.067,
            0.067,
            0.067,
            0.025,
            0.067,
            0.067,
            0.067,
            0.067,
            0.035,
            0.067,
            0.035,
            0.067,
            0.067,
            0.035,
            0.067,
            0.035,
            0.067,
            0.079,
            0.072,
            0.062,
            0.079,
            0.072,
            0.062,
            0.089,
            0.107,
            0.107,
            0.087,
            0.087,
            0.089,
            0.067,
            0.067,
            0.067
        ],
        skeleton_info={
            "0": {
                "link": [
                    "left_eye",
                    "right_eye"
                ],
                "id": 0,
                "color": [
                    255.0,
                    198.17042843358632,
                    0.0
                ]
            },
            "1": {
                "link": [
                    "left_eye",
                    "left_earbase"
                ],
                "id": 1,
                "color": [
                    74.02505744611004,
                    255.0,
                    0.0
                ]
            },
            "2": {
                "link": [
                    "left_earbase",
                    "left_earend"
                ],
                "id": 2,
                "color": [
                    34.390971759392784,
                    255.0,
                    0.0
                ]
            },
            "3": {
                "link": [
                    "right_eye",
                    "right_earbase"
                ],
                "id": 3,
                "color": [
                    255.0,
                    237.8045141203036,
                    0.0
                ]
            },
            "4": {
                "link": [
                    "right_earbase",
                    "right_earend"
                ],
                "id": 4,
                "color": [
                    232.56140019297916,
                    255.0,
                    0.0
                ]
            },
            "5": {
                "link": [
                    "left_eye",
                    "nose"
                ],
                "id": 5,
                "color": [
                    113.65914313282731,
                    255.0,
                    0.0
                ]
            },
            "6": {
                "link": [
                    "lower_jaw",
                    "mouth_end_left"
                ],
                "id": 6,
                "color": [
                    255.0,
                    158.53634274686905,
                    0.0
                ]
            },
            "7": {
                "link": [
                    "lower_jaw",
                    "mouth_end_right"
                ],
                "id": 7,
                "color": [
                    255.0,
                    118.9022570601518,
                    0.0
                ]
            },
            "8": {
                "link": [
                    "lower_jaw",
                    "upper_jaw"
                ],
                "id": 8,
                "color": [
                    255.0,
                    79.26817137343453,
                    0.0
                ]
            },
            "9": {
                "link": [
                    "right_eye",
                    "nose"
                ],
                "id": 9,
                "color": [
                    255.0,
                    198.17042843358632,
                    0.0
                ]
            },
            "10": {
                "link": [
                    "back_base",
                    "back_middle"
                ],
                "id": 10,
                "color": [
                    0.0,
                    255.0,
                    243.04619726140453
                ]
            },
            "11": {
                "link": [
                    "back_middle",
                    "back_end"
                ],
                "id": 11,
                "color": [
                    0,
                    255,
                    255
                ]
            },
            "12": {
                "link": [
                    "back_end",
                    "tail_base"
                ],
                "id": 12,
                "color": [
                    0,
                    220,
                    255
                ]
            },
            "13": {
                "link": [
                    "back_middle",
                    "body_middle_left"
                ],
                "id": 13,
                "color": [
                    255.0,
                    0.0,
                    23.90625
                ]
            },
            "14": {
                "link": [
                    "back_middle",
                    "body_middle_right"
                ],
                "id": 14,
                "color": [
                    255.0,
                    0.0,
                    63.54033568671722
                ]
            },
            "15": {
                "link": [
                    "body_middle_right",
                    "belly_bottom"
                ],
                "id": 15,
                "color": [
                    255.0,
                    0.0,
                    63.54033568671722
                ]
            },
            "16": {
                "link": [
                    "body_middle_left",
                    "belly_bottom"
                ],
                "id": 16,
                "color": [
                    255.0,
                    0.0,
                    23.90625
                ]
            },
            "17": {
                "link": [
                    "tail_base",
                    "tail_end"
                ],
                "id": 17,
                "color": [
                    0,
                    165,
                    255
                ]
            },
            "18": {
                "link": [
                    "throat_end",
                    "front_left_thai"
                ],
                "id": 18,
                "color": [
                    0.0,
                    255.0,
                    203.4123612681037
                ]
            },
            "19": {
                "link": [
                    "front_left_thai",
                    "front_left_knee"
                ],
                "id": 19,
                "color": [
                    0.0,
                    68.78344961404169,
                    255.0
                ]
            },
            "20": {
                "link": [
                    "front_left_knee",
                    "front_left_paw"
                ],
                "id": 20,
                "color": [
                    0.0,
                    29.14936392732455,
                    255.0
                ]
            },
            "21": {
                "link": [
                    "throat_end",
                    "front_right_thai"
                ],
                "id": 21,
                "color": [
                    0.0,
                    255.0,
                    203.4123612681037
                ]
            },
            "22": {
                "link": [
                    "front_right_thai",
                    "front_right_knee"
                ],
                "id": 22,
                "color": [
                    89.75289313282732,
                    0.0,
                    255.0
                ]
            },
            "23": {
                "link": [
                    "front_right_knee",
                    "front_right_paw"
                ],
                "id": 23,
                "color": [
                    89.75289313282732,
                    0.0,
                    255.0
                ]
            },
            "24": {
                "link": [
                    "tail_base",
                    "back_left_thai"
                ],
                "id": 24,
                "color": [
                    0,
                    165,
                    255
                ]
            },
            "25": {
                "link": [
                    "back_left_thai",
                    "back_left_knee"
                ],
                "id": 25,
                "color": [
                    169.02106450626192,
                    0.0,
                    255.0
                ]
            },
            "26": {
                "link": [
                    "back_left_knee",
                    "back_left_paw"
                ],
                "id": 26,
                "color": [
                    169.02106450626192,
                    0.0,
                    255.0
                ]
            },
            "27": {
                "link": [
                    "tail_base",
                    "back_right_thai"
                ],
                "id": 27,
                "color": [
                    0,
                    165,
                    255
                ]
            },
            "28": {
                "link": [
                    "back_right_thai",
                    "back_right_knee"
                ],
                "id": 28,
                "color": [
                    255.0,
                    0.0,
                    142.80850706015173
                ]
            },
            "29": {
                "link": [
                    "back_right_knee",
                    "back_right_paw"
                ],
                "id": 29,
                "color": [
                    255.0,
                    0.0,
                    142.80850706015173
                ]
            },
            "30": {
                "link": [
                    "left_antler_base",
                    "left_antler_end"
                ],
                "id": 30,
                "color": [
                    0.0,
                    255.0,
                    44.87701729490043
                ]
            },
            "31": {
                "link": [
                    "right_antler_base",
                    "right_antler_end"
                ],
                "id": 31,
                "color": [
                    153.2932288195446,
                    255.0,
                    0.0
                ]
            },
            "32": {
                "link": [
                    "neck_base",
                    "neck_end"
                ],
                "id": 32,
                "color": [
                    0.0,
                    255.0,
                    84.51085328820125
                ]
            },
            "33": {
                "link": [
                    "neck_end",
                    "back_base"
                ],
                "id": 33,
                "color": [
                    0.0,
                    255.0,
                    124.14468928150207
                ]
            },
            "34": {
                "link": [
                    "back_base",
                    "front_right_thai"
                ],
                "id": 34,
                "color": [
                    0.0,
                    255.0,
                    243.04619726140453
                ]
            },
            "35": {
                "link": [
                    "back_base",
                    "front_left_thai"
                ],
                "id": 35,
                "color": [
                    0.0,
                    255.0,
                    243.04619726140453
                ]
            },
            "36": {
                "link": [
                    "throat_base",
                    "throat_end"
                ],
                "id": 36,
                "color": [
                    0.0,
                    255.0,
                    163.77852527480275
                ]
            },
            "37": {
                "link": [
                    "belly_bottom",
                    "body_middle_left"
                ],
                "id": 37,
                "color": [
                    255.0,
                    0.0,
                    23.90625
                ]
            },
            "38": {
                "link": [
                    "belly_bottom",
                    "body_middle_right"
                ],
                "id": 38,
                "color": [
                    255.0,
                    0.0,
                    63.54033568671722
                ]
            },
            "39": {
                "link": [
                    "left_earbase",
                    "right_earbase"
                ],
                "id": 39,
                "color": [
                    74.02505744611004,
                    255.0,
                    0.0
                ]
            }
        }
)