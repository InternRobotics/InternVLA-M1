


def build_framework(cfg):

    if cfg.framework.framework_py == "qwendino_cogactheader":
        from llavavla.model.framework.qwendino_cogactheader import build_model_framework

        return build_model_framework(cfg)

