


def build_framework(cfg):

    if cfg.framework.framework_py == "InternVLA-M1":
        from InternVLA.model.framework.M1 import build_model_framework

        return build_model_framework(cfg)
    
    raise NotImplementedError(f"Framework {cfg.framework.framework_py} is not implemented.")

