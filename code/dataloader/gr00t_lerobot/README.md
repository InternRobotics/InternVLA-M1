# add by @JinhuiYE
# 结构

1. ./data_config.py
* 定义了不同数据集的 内部key:value 情况
* 和数据记录的 ../meta/modality.json 呼应
2. ./*/mixtures.py
* 使用别名表示不同的数据集配比组合

3. ./transform