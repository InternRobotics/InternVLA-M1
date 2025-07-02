import numpy as np
import time

from eval.sim_cogact.qwenact_qformer_policy import QwenACTAFormerInference


class ModelController:
    def __init__(self, model_path=None, unnorm_key=None, debug=False, debug_path=None):
        if debug:
            self.actions = np.load(debug_path)
            self.model = QwenACTAFormerInference(saved_model_path=model_path, unnorm_key=unnorm_key)
            # time.sleep(20)
        else:
            self.model = QwenACTAFormerInference(saved_model_path=model_path, unnorm_key=unnorm_key)

    
    def infer_debug_debug(self, t, rgb_images, lang: str):
        # _ = self.model.step(rgb_images, lang)
        return self.actions[t]



    def infer(self, rgb_images, lang: str):
        action_chunk = self.model.step(rgb_images, lang)
        return action_chunk[0]
    
    def infer_debug(self, t):
        return self.actions[t]
