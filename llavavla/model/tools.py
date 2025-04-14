def auto_get_module_keys(module, max_depth=0, prefix_list=None, current_depth=0, current_prefix=''):
    """
    获取模块的所有子模块键，支持设置递归深度和前缀列表。

    :param module: 要遍历的PyTorch模块。
    :param max_depth: 最大递归深度，默认为1。
    :param prefix_list: 仅包含指定前缀的模块，默认为None表示不限制。
    :param current_depth: 当前递归深度，内部使用。
    :param current_prefix: 当前前缀，内部使用。
    :return: 模块键的列表。
    """
    if current_depth > max_depth:
        return []

    module_keys = []
    for name, sub_module in module.named_children():
        full_name = f"{current_prefix}.{name}" if current_prefix else name
        if prefix_list is None or any(full_name.startswith(prefix) for prefix in prefix_list):
            module_keys.append(full_name)
        module_keys.extend(auto_get_module_keys(sub_module, max_depth, prefix_list, current_depth + 1, full_name))
    return module_keys


import torch.nn as nn

def is_module_trainable(module):
    """
    判断一个模块是否可训练：如果模块本身有参数，则要求其所有参数 require_grad 为 True；
    如果模块没有直接参数，则认为其是否可训练依赖于子模块。
    """
    params = list(module.parameters(recurse=False))
    if params:
        return all(p.requires_grad for p in params)
    else:
        # 对于没有直接参数的容器模块，视为可训练（最终结果取决于其子模块）
        return True

def auto_get_trainable_modules(module, prefix='', max_depth=None):
    """
    递归遍历模块，返回所有可训练模块的名称列表。
    如果一个模块的所有子模块都是可训练的，则只返回父模块的名称，不再递归输出其各个子模块名称。
    
    参数：
      - module: 需要遍历的模块。
      - prefix: 当前模块的名称前缀（内部使用）。
      - max_depth: 最大递归深度，None 表示无限递归。
      
    返回：
      一个模块名称的列表。
    """
    # 获取当前模块的所有直接子模块
    children = list(module.named_children())
    
    # 如果达到最大深度或没有子模块，则返回当前模块（如果可训练且prefix非空）
    if (max_depth is not None and max_depth <= 0) or not children:
        return [prefix] if prefix and is_module_trainable(module) else []
    
    child_keys = []
    all_children_trainable = True
    for name, child in children:
        full_name = f"{prefix}.{name}" if prefix else name
        # 递归获取子模块的可训练键
        keys = auto_get_trainable_modules(child, full_name, None if max_depth is None else max_depth - 1)
        if not keys:
            # 如果子模块没有进一步的子模块返回，则检查子模块自身
            if is_module_trainable(child):
                keys = [full_name]
            else:
                all_children_trainable = False
        else:
            # 如果子模块返回了多个名称，则说明未能合并
            if len(keys) > 1:
                all_children_trainable = False
        child_keys.extend(keys)
    
    # 如果当前模块可训练且所有子模块都可训练，则返回当前模块名称
    if is_module_trainable(module) and all_children_trainable and child_keys:
        return [prefix] if prefix else child_keys
    else:
        return child_keys

    # 示例用法：假设 cogact_qwen 是一个包含 vlm 和 action_model 的模型实例
    # condensed_keys = get_condensed_trainable_modules(cogact_qwen, max_depth=2)
    # print(condensed_keys)
    #     # 读取配置文件