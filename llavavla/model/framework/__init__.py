


def build_framework(cfg):

    if cfg.framework.framework_py == "DinoQFormerACT":
        from llavavla.model.framework.DinoQFormerACT import build_model_framework

        return build_model_framework(cfg)

    elif cfg.framework.framework_py == "qwenact":
        from llavavla.model.framework.qwenact import build_model_framework

        return build_model_framework(cfg)
    elif cfg.framework.framework_py == "qwenpi":
        from llavavla.model.framework.qwenpi import build_model_framework

        return build_model_framework(cfg)
    
    
    