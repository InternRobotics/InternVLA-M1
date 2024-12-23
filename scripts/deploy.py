"""
deploy.py

A simple deployment example for serving a fine-tuned model. 
Please execute the code below to start the server. Using 'fractal20220817_data' as an example, 
please replace "unnorm_key" with the value from your fine-tuned dataset in actual use.

```
python scripts/deploy.py --saved_model_path <your_model_path> --unnorm_key fractal20220817_data --action_ensemble --use_bf16 --action_ensemble_horizon 2 --adaptive_ensemble_alpha 0.1 --cfg_scale 1.5 --port 5500

```


The client only needs a Python environment and the requests library (pip install requests); 
no other dependencies need to be installed.

Client (Standalone) Usage (assuming a server running on 0.0.0.0:5500):

```
import requests
import json

# Define the API endpoint
url = 'http://127.0.0.1:5500/api/inference'

# Define the parameters you want to send
data = {
    'task_description': "Pick up the red can.",
}
image = "image/google_robot.png"

json.dump(data, open("data.json", "w"))

with open ("data.json", "r") as query_file:
    with open(image, "rb") as image_file:
        file = [
            ('images', (image, image_file, 'image/png')),
            ('json', ("data.json", query_file, 'application/json'))
        ]
        
        response = requests.post(url, files=file)
if response.status_code == 200:
    pass
else:
    print("Failed to get a response from the API")
    print(response.text)
```

"""

import numpy as np
from PIL import Image
from typing import Optional, Tuple, Union
import os
import argparse
import json
import math
from flask import Flask, request, jsonify
import tempfile
import torch
from vla import load_vla
from sim_cogact.adaptive_ensemble import AdaptiveEnsembler

app = Flask(__name__)


class CogACTService:
    def __init__(
        self,
        saved_model_path: str = "CogACT/CogACT-Base",
        unnorm_key: str = None,
        image_size: list[int] = [224, 224],
        action_model_type: str = "DiT-B",  # choose from ['DiT-Small', 'DiT-Base', 'DiT-Large'] to match the model weight
        future_action_window_size: int = 15,
        cfg_scale: float = 1.5,
        num_ddim_steps: int = 10, 
        use_ddim: bool = True,
        use_bf16: bool = True,
        action_dim: int = 7,
        action_ensemble: bool = True,
        adaptive_ensemble_alpha: float = 0.1,
        action_ensemble_horizon: int = 2,
        action_chunking: bool = False,
        action_chunking_window: Optional[int] = None,
        args=None
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        assert not (action_chunking and action_ensemble), "Now 'action_chunking' and 'action_ensemble' cannot both be True."  

        self.unnorm_key = unnorm_key

        print(f"*** unnorm_key: {unnorm_key} ***")
        self.vla = load_vla(
          saved_model_path,
          load_for_training=False, 
          action_model_type=action_model_type,
          future_action_window_size=future_action_window_size,
          action_dim=action_dim,
        )
        if use_bf16:
            self.vla.vlm = self.vla.vlm.to(torch.bfloat16)
        self.vla = self.vla.to("cuda").eval()
        self.cfg_scale = cfg_scale

        self.image_size = image_size
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.action_chunking = action_chunking
        self.action_chunking_window = action_chunking_window
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None

        self.args = args
        self.reset()

    def reset(self) -> None:
        if self.action_ensemble:
            self.action_ensembler.reset()

    def step(
        self, image: str, 
        task_description: Optional[str] = None, 
        *args, **kwargs,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: Path to the image file
            task_description: Optional[str], task description
        Output:
            action: list[float], the ensembled 7-DoFs action of End-effector and gripper

        """

        image: Image.Image = Image.open(image)

        # [IMPORTANT!]: Please process the input images here in exactly the same way as the images
        # were processed during finetuning to ensure alignment between inference and training.
        # Make sure, as much as possible, that the gripper is visible in the processed images.
        resized_image = resize_image(image, size=self.image_size)
        unnormed_actions, normalized_actions = self.vla.predict_action(
            image=resized_image, 
            instruction=task_description, 
            unnorm_key=self.unnorm_key, 
            do_sample=False, 
            cfg_scale=self.cfg_scale, 
            use_ddim=self.use_ddim, 
            num_ddim_steps=self.num_ddim_steps,
            )

        if self.action_ensemble:
            unnormed_actions = self.action_ensembler.ensemble_action(unnormed_actions)
            # Translate the value of the gripper's open/close state to 0 or 1.
            # Please adjust this line according to the control mode of different grippers.
            unnormed_actions[6] = unnormed_actions[6] > 0.5
            action = unnormed_actions.tolist()
        elif self.action_chunking:
            # [IMPORTANT!]: Please modify the code here to output multiple actions at once.
            # The code below only outputs the first action in the chunking.
            # The chunking window size can be adjusted by modifying the 'action_chunking_window' parameter.
            if self.action_chunking_window is not None:
                chunked_actions = []
                for i in range(0, self.action_chunking_window):
                    chunked_actions.append(unnormed_actions[i].tolist())
                action = chunked_actions
            else:
                raise ValueError("Please specify the 'action_chunking_window' when using action chunking.")
        else:
            # Output the first action in the chunking. Can be modified to output multiple actions at once.
            unnormed_actions = unnormed_actions[0]
            action = unnormed_actions.tolist()

        print(f"Instruction: {task_description}")
        print(f"Model path: {self.args.saved_model_path} at port {self.args.port}")
        return action


# [IMPORTANT!]: Please modify the image processing code here to ensure that the input images  
# are handled in exactly the same way as during the finetuning phase.
# Make sure, as much as possible, that the gripper is visible in the processed images.
def resize_image(image: Image, size=(224, 224), shift_to_left=0):
    w, h = image.size
    assert h < w, "Height should be less than width"
    left_margin = (w - h) // 2 - shift_to_left
    left_margin = min(max(left_margin, 0), w - h)
    image = image.crop((left_margin, 0, left_margin + h, h))

    image = image.resize(size, resample=Image.LANCZOS)
    
    image = scale_and_resize(image, target_size=(224, 224), scale=0.9, margin_w_ratio=0.5, margin_h_ratio=0.5)
    return image

# Here the image is first center cropped and then resized back to its original size 
# because random crop data augmentation was used during finetuning.
def scale_and_resize(image : Image, target_size=(224, 224), scale=0.9, margin_w_ratio=0.5, margin_h_ratio=0.5):
    w, h = image.size
    new_w = int(w * math.sqrt(scale))
    new_h = int(h * math.sqrt(scale))
    margin_w_max = w - new_w
    margin_h_max = h - new_h
    margin_w = int(margin_w_max * margin_w_ratio)
    margin_h = int(margin_h_max * margin_h_ratio)
    image = image.crop((margin_w, margin_h, margin_w + new_w, margin_h + new_h))
    image = image.resize(target_size, resample=Image.LANCZOS)
    return image


parser = argparse.ArgumentParser()
parser.add_argument("--saved_model_path", type=str, default="CogACT/CogACT-Base")
parser.add_argument("--unnorm_key", type=str, default=None)
parser.add_argument("--image_size", type=list[int], default=[224, 224])
parser.add_argument("--action_model_type", type=str, default="DiT-B")
parser.add_argument("--future_action_window_size", type=int, default=15)
parser.add_argument("--cfg_scale", type=float, default=1.5)
parser.add_argument("--port", type=int, default=5500)
parser.add_argument("--use_bf16", action="store_true")
parser.add_argument("--action_dim", type=int, default=7)
parser.add_argument("--action_ensemble", action="store_true")
parser.add_argument("--action_ensemble_horizon", type=int, default=2)
parser.add_argument("--adaptive_ensemble_alpha", type=float, default=0.1)
parser.add_argument("--action_chunking", action="store_true")
parser.add_argument("--action_chunking_window", type=int, default=None)

args = parser.parse_args()

inferencer = CogACTService(
    saved_model_path=args.saved_model_path,
    unnorm_key=args.unnorm_key,
    image_size=args.image_size,
    action_model_type=args.action_model_type,
    future_action_window_size=args.future_action_window_size,
    cfg_scale=args.cfg_scale,
    use_bf16=args.use_bf16,
    action_dim=args.action_dim,
    action_ensemble=args.action_ensemble,
    adaptive_ensemble_alpha=args.adaptive_ensemble_alpha,
    action_ensemble_horizon=args.action_ensemble_horizon,
    action_chunking=args.action_chunking,
    action_chunking_window=args.action_chunking_window,
    args=args
)

@app.route('/api/inference', methods=['POST'])
def inference():
    image = request.files['images']
    query = request.files['json']
    with tempfile.NamedTemporaryFile(delete=False) as temp_image:
        image.save(temp_image.name)
        temp_image_path = temp_image.name
    with tempfile.NamedTemporaryFile(delete=False) as temp_query:
        query.save(temp_query.name)
        temp_query_path = temp_query.name
    input_query = json.load(open(temp_query_path))
    print(input_query)
    answer = inferencer.step(temp_image_path, **input_query)
    # clean files
    os.remove(temp_image_path)
    os.remove(temp_query_path)
    return jsonify(answer)

if __name__ == "__main__":
    
    app.run(host="0.0.0.0", debug=False, port=args.port)