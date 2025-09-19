import os

# from IPython import embed; embed()
from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from examples.Eval_simplenv_clean.m1_server_simpler_env import M1Inference

import numpy as np

# try:
#     from simpler_env.policies.octo.octo_model import OctoInference
# except ImportError as e:
#     print("Octo is not correctly imported.")
#     print(e)

if os.environ.get("DEBUG", None):
    import debugpy
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"







if __name__ == "__main__":
    args = get_args()

    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    import debugpy 

    # debugpy.listen(("0.0.0.0", 10092))  # listen port 
    # print("Waiting for debugger to attach...")
    # debugpy.wait_for_client()  # wait for VS Code attach



    model = M1Inference(
        policy_setup=args.policy_setup,
        action_scale=args.action_scale,
        cfg_scale=1.5                     # cfg from 1.5 to 7 also performs well
    )

    # policy model creation; update this if you are using a new policy model
    # run real-to-sim evaluation
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
