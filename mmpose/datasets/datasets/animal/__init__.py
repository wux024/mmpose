# Copyright (c) OpenMMLab. All rights reserved.
from .animalkingdom_dataset import AnimalKingdomDataset
from .animalpose_dataset import AnimalPoseDataset
from .ap10k_dataset import AP10KDataset
from .atrw_dataset import ATRWDataset
from .fly_dataset import FlyDataset
from .horse10_dataset import Horse10Dataset
from .locust_dataset import LocustDataset
from .macaque_dataset import MacaqueDataset
from .zebra_dataset import ZebraDataset
from .acinoset_dataset import AcinoSetDataset
from .aniposefly_dataset import AniposeFlyDataset
from .aniposemouse_dataset import AniposeMouseDataset
from .apt36k_dataset import APT36KDataset
from .aptv2_dataset import APTv2Dataset
from .awa_dataset import AWADataset
from .fish_dataset import FishDataset
from .lote_dataset import LoTEDataset
from .marmoset_dataset import MarmosetDataset
from .mouse_dataset import MouseDataset
from .openmonkeychallenge_dataset import OpenMonkeyChallengeDataset
from .pups_dataset import PupsDataset
from .quadruped_dataset import QuadrupedDataset
from .stanfordextra_dataset import StanfordExtraDataset
from .topviewmouse_dataset import TopViewMouseDataset


__all__ = [
    'AnimalPoseDataset', 'AP10KDataset', 'Horse10Dataset', 'MacaqueDataset',
    'FlyDataset', 'LocustDataset', 'ZebraDataset', 'ATRWDataset',
    'AnimalKingdomDataset', 'AcinoSetDataset', 'AniposeFlyDataset',
    'AniposeMouseDataset', 'APT36KDataset', 'APTv2Dataset', 'AWADataset',
    'FishDataset', 'LoTEDataset', 'MarmosetDataset', 'MouseDataset',
    'OpenMonkeyChallengeDataset', 'PupsDataset', 'QuadrupedDataset',
    'StanfordExtraDataset', 'TopViewMouseDataset'
]
