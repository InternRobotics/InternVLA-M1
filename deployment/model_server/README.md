
# start policy server


```bash

your_ckpt=results/Checkpoints/1_need/0906_bestvla_retrain_sota2/checkpoints/steps_50000_pytorch_model.pt

python deployment/model_server/server_policy_M1.py \
    --ckpt_path ${your_ckpt} \
    --port 10093 \
    --use_bf16
```


# connect to policy server

```python
# in your sim <-> vla controler
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
client = WebsocketClientPolicy(host, port)
```