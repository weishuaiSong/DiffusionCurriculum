# DiffusionCurriculum

## QA API 调用
参见`scripts/test_api.py`

Note: 如果需要解析llm输出的json格式结果，可以使用：
```python
from src.utils import JSONParser

results = JSONParser.parse("your response")
```