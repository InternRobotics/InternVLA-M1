


def build_framework(cfg):

    if cfg.framework.framework_py == "DinoQFormerACT":
        from llavavla.model.framework.InternVLA import build_model_framework

        return build_model_framework(cfg)

    elif cfg.framework.framework_py == "qwenact":
        from llavavla.model.framework.qwenact import build_model_framework

        return build_model_framework(cfg)
    
    elif cfg.framework.framework_py == "qwenpi":
        from llavavla.model.framework.qwenpi import build_model_framework

        return build_model_framework(cfg)

    
    elif cfg.framework.framework_py == "qwendino_plus":
        from llavavla.model.framework.qwendino_plus import build_model_framework

        return build_model_framework(cfg)
    
    