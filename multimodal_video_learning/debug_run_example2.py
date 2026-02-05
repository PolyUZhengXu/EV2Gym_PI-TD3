import traceback
from multimodal_video_learning import examples

try:
    examples.example_2_feature_extraction()
except Exception:
    traceback.print_exc()
    raise
