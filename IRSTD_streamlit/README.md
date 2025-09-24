# 说明文档

> 红外小目标检测系统

## 🎉 特性

- 使用 Steamlit 构建 Web 页面

## 🐋 依赖

> 推荐使用 pipenv 新建一个虚拟环境来管理 pip 包，防止依赖冲突。具体使用方法见 [Pipenv 使用说明] 。

```bash
# 使用 pipenv 安装依赖并创建虚拟环境
# 安装完成后需要在 VS Code 中选择虚拟环境
pipenv install
pipenv install --dev

# 使用 pip 直接在本地环境安装依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## 🚀 运行

```bash
# 启动服务
python -m streamlit run gui/home.py

# 或选择在 8081 端口上启动服务
python -m streamlit run gui/home.py --server.port 8081
```

## ⚙️ 配置服务

修改 Streamlit 服务 [配置文件]

## 📦 添加模型

在模型包的 [声明文件] 中的 `ModelType` 枚举类中添加新的枚举项：

```python
class ModelType(Enum):
    LW_IRST_ablation = "./LW_IRST.onnx"
    NEW_MODEL = "./path_to_model"  # 声明新的枚举项，值为模型的存放位置
```

## 需要明确

- 模型评估中没有找到加载数据集的地方
-

## 📄 相关文档

- [Steamlit API reference]

<!-- Links -->

[配置文件]: .streamlit\config.toml
[声明文件]: .models/__init__.py

[Pipenv 使用说明]: ./docs/pipenv-useages.md

[Steamlit API reference]: https://docs.streamlit.io/library/api-reference
