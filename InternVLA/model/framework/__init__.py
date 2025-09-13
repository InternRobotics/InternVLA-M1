


def build_framework(cfg):

    if cfg.framework.framework_py == "qwendino_cogactheader":
        from InternVLA.model.framework.M1 import build_model_framework

        return build_model_framework(cfg)

