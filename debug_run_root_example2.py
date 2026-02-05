import sys, traceback
sys.path.insert(0, r'D:\Program Files\PolyUCode\EV2Gym_PI-TD3')
from multimodal_video_learning import examples

try:
    examples.example_2_feature_extraction()
except Exception:
    traceback.print_exc()
    raise
