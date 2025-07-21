import numpy as np
import time
import os
import torch
import pynput
from datetime import datetime
from loguru import logger
from scipy.spatial.transform import Rotation as R
import numpy as np
from sanic import Sanic, response


from embodieddeploy.controller.franka_pandapy_impedance import FrankaController
from embodieddeploy.sensor.camera.realsense import MultiRealSenseCamera
from real_deployment.controller.utils import adjust_translation_along_quaternion, RT_to_tran_quaternion, set_seed, trans_quant_to_RT
from real_deployment.controller.controller import ModelController
from PIL import Image

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# ... existing imports ...



status = "ok"
want_to_pause = False
def on_press(key):
    global status
    global want_to_pause
    if key == pynput.keyboard.Key.space:
        status = "pause" if status == "ok" else "ok"
        want_to_pause = not want_to_pause
    elif str(key) == "p":
        status = "pause" if status == "ok" else "ok"
        want_to_pause = not want_to_pause
    elif key == pynput.keyboard.Key.esc:
        status = "end"
    elif str(key) == "'r'":
        status = "reset"
    return





class AgentSingleton:
    _instance = None
    model_path = ""
    unnorm_key = ""

    @staticmethod
    def set_config(model_path: str, unnorm_key: str):
        AgentSingleton.model_path = model_path
        AgentSingleton.unnorm_key = unnorm_key


    @staticmethod
    def get_instance() -> ModelController:
        if AgentSingleton._instance is None:
            model_path = AgentSingleton.model_path
            unnorm_key = AgentSingleton.unnorm_key
            AgentSingleton._instance = ModelController(model_path=model_path, unnorm_key=unnorm_key)
        return AgentSingleton._instance




class CameraSingleton:
    _instance = None

    @staticmethod
    def get_instance():
        if CameraSingleton._instance is None:
            CameraSingleton._instance = MultiRealSenseCamera(image_height=480, image_width=640, fps=60)
            for _ in range(10):
                images = CameraSingleton._instance.undistorted_rgb()
        return CameraSingleton._instance
        
    
    
    def get_images(self):
        images = self._instance.undistorted_rgb()
        return images[0]


app = Sanic("InferenceServer")

@app.post("/infer")
async def infer(request):
    camera = CameraSingleton.get_instance()

    model_path = "/data/share/yujunqiu/0628/checkpoints/steps_20000_pytorch_model.pt"
    unnorm_key = "place_0627_lmdb"
    AgentSingleton.set_config(model_path, unnorm_key)
    agent :ModelController = AgentSingleton.get_instance()
        
    images = camera.undistorted_rgb()

    assert len(images) == 2, "only two cameras are supported"
    lang = "Place the cucumber on the plate"

    action = agent.infer(images, lang)
    logger.debug(f"actions: {action}")

    return response.json({"action": action.tolist()})



@app.post("/infer")
async def infer(request):
    camera = CameraSingleton.get_instance()

    model_path = "/data/yujunqiu/code/llavavla/ckpts/0704_20_place/checkpoints/steps_12000_pytorch_model.pt"
    unnorm_key = "place_0704_2"
    AgentSingleton.set_config(model_path, unnorm_key)
    agent :ModelController = AgentSingleton.get_instance()


    data = request.json
    timestep = data['t']

    images = camera.undistorted_rgb()

    assert len(images) == 2, "only two cameras are supported"
    
    # for i in range(2):
    #     Image.fromarray(images[i]).save(f"debug/infer_images/image_{timestep:04d}_{i}.png")
    
    lang = "Place the orange carrot on the plate"

    action = agent.infer(images, lang)
    logger.debug(f"actions: {action}")

    return response.json({"action": action.tolist()})



@app.post("/debug")
async def debug(request):
    camera = CameraSingleton.get_instance()

    model_path = "/data/share/yujunqiu/0628/checkpoints/steps_20000_pytorch_model.pt"
    unnorm_key = "place_0627_lmdb"
    AgentSingleton.set_config(model_path, unnorm_key)
    agent :ModelController = AgentSingleton.get_instance()
        
    images = camera.undistorted_rgb()

    assert len(images) == 2, "only two cameras are supported"
    lang = "Place the cucumber on the plate"

    action = agent.infer(images, lang)
    logger.debug(f"actions: {action}")

    return response.json({"action": action.tolist()})




def main():
    app.run(host="0.0.0.0", port=15252)



if __name__ == "__main__":
    main()