"""多模态特征提取和融合模块"""

from .vision_extractor import VisionFeatureExtractor
from .audio_extractor import AudioFeatureExtractor
from .text_extractor import TextFeatureExtractor
from .fusion_module import DynamicMultimodalFusion
from .alignment_module import FeatureAlignment
from .temporal_module import TemporalSemanticModule

__all__ = [
    'VisionFeatureExtractor',
    'AudioFeatureExtractor',
    'TextFeatureExtractor',
    'DynamicMultimodalFusion',
    'FeatureAlignment',
    'TemporalSemanticModule',
]
