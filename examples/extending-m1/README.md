
## 🧩 Extending InternVLA-M1

### 1. Data Format & Loading

Our data format follows best practices from open-source VLA efforts.

* Action data follows the  format provided by  (see [https://github.com/NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)).
* Multimodal data follows the structure used in  ([https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune](https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune)).

To validate your dataloader:

```bash
python ./InternVLA/dataloader/lerobot_datasets_oxe.py
python ./InternVLA/dataloader/vlm_datasets.py
```

If the returned samples match your expectations, register your custom dataloader in:
`InternVLA/dataloader/__init__.py`

---

### 2. Model Development

To add a new framework:

* Implement your model in a single `framework.py` file (e.g. `InternVLA/model/framework/M1.py`), aligned with your paper’s framework figure.
* Define required modules in `InternVLA/model/modules/`.
* Ensure you can run:

  ```bash
  python InternVLA/model/framework/M1.py
  ```

  and build your model to run a forward pass on a batch sample.
* Register your framework in:
  `InternVLA/model/framework/__init__.py`

---

### 3. Model Deployment

All frameworks can be loaded and used as follows:

```python
from InternVLA.model.framework.yourmodel import YourFramework

ckpt = "playground/Checkpoints/debug_0910_internM1_cotrain/checkpoints/steps_2000_pytorch_model.pt"
model = YourFramework.from_pretrained(ckpt)

model.predict_action(...)
```

You can also serve models via the deployment service under `deployment/model_server`.

---

### 4. Configuration

InternVLA-M1 uses a single **global YAML configuration** file to manage all parameters.

* The default config is at:
  `InternVLA/config/training/qwenvla_cotrain_oxe.yaml`
* It acts as a flexible dictionary-like object. Parameters can be redundant (unused fields are allowed) but must not be missing if referenced.
* Parameters are organized into groups: `datasets`, `framework`, `trainer`.
* You can override or add parameters via command line arguments.
* Final resolved configs are saved in the checkpoint folder for reproducibility.

