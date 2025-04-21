# DiffusionCurriculum

## Install
```
git clone https://github.com/shijian2001/DiffusionCurriculum
cd DiffusionCurriculum
pip install -e .
```
`Note：必须按照上述方法安装，不然无法正常使用我们自己开发的package`

## QA API 调用
参见`scripts/test_api.py`

Note: 如果需要解析llm输出的json格式结果，可以使用：
```python
from src.utils import JSONParser

results = JSONParser.parse("your response")
```
如果想检验自己输出的格式或者有更复杂的验证逻辑，可以定义`validate_func(answer:str)->Any`,若没有则传入`None`。其中validate函数的返回应该是通过验证的结果或False。切记不能返回True:
```python
async for result in generator.generate_stream(prompts, system_prompt, validate_func=validate_func):
    pass
```

## Dev Process
参照`relation generation`的开发流程：
1. 在`src/scene_graph_builder/relation_generator.py`中开发核心功能
2. 在`scripts/generate_relations`开发脚本
3. 脚本传参完全通过`configs/relation_gen.yaml`控制，执行:
```
python scripts/generate_relations.py configs/relation_gen.yaml
```
如果你有新添加的package，请在`pyproject.toml`中的dependence中添加：
```
dependencies = [
    "openai>=0.28.0",
    "PyYAML>=6.0",
    ... # Add
]
```