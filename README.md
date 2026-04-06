# RustEpubReader-Model

中文拼写纠错 (CSC) 模型仓库 — [RustEpubReader](https://github.com/zhongbai2333/RustEpubReader) 的 AI 模型管线。

## 仓库结构

```
├── .github/workflows/     # CI/CD 流水线
│   ├── build-model.yml    # 模型导出 + 量化 + 发布 Release
│   ├── train.yml          # 触发 PAI-DLC 微调训练
│   └── validate-pr.yml    # 自动校验社区数据提交 PR
├── config/
│   └── training_config.yaml  # 训练超参数
├── datasets/
│   ├── README.md          # 数据集贡献指南
│   ├── base/              # 基础训练集 (SIGHAN + 公开数据)
│   └── submissions/       # 社区贡献的纠错数据
├── scripts/
│   ├── export_onnx.py     # PyTorch → ONNX 导出
│   ├── quantize.py        # ONNX 动态 INT8 量化
│   ├── train.py           # LoRA 微调训练脚本
│   ├── generate_manifest.py   # 生成 manifest.json + SHA256
│   └── validate_submissions.py # 数据提交格式校验
└── requirements.txt       # Python 依赖
```

## 模型信息

| 项目 | 说明 |
|---|---|
| 基础模型 | [shibing624/macbert4csc-base-chinese](https://huggingface.co/shibing624/macbert4csc-base-chinese) |
| 任务 | 中文拼写纠错 (Chinese Spelling Correction) |
| 量化 | ONNX 动态 INT8 量化 (~100MB) |
| 推理框架 | ONNX Runtime (ort crate) |
| 分词器 | BERT WordPiece (`vocab.txt`) |
| 最大序列长度 | 128 tokens |

## 快速开始

### 1. 手动构建模型

```bash
pip install -r requirements.txt

# 导出 ONNX
python scripts/export_onnx.py

# INT8 量化
python scripts/quantize.py

# 生成 manifest
python scripts/generate_manifest.py
```

产物保存在 `output/` 目录：
- `csc-macbert-int8.onnx` — 量化后的 ONNX 模型
- `csc-vocab.txt` — BERT 词表
- `csc-manifest.json` — 版本清单 + SHA256 校验和

### 2. 使用 CI 自动构建

推送 tag `v*` 即可触发自动构建和 Release 发布：

```bash
git tag v1.0.0
git push origin v1.0.0
```

GitHub Actions 会自动：
1. 下载基础模型 → 导出 ONNX → INT8 量化
2. 生成 manifest.json (含 SHA256)
3. 创建 GitHub Release 并上传所有产物
4. CDN (`dl.zhongbai233.com`) 自动从 Release 同步

### 3. 微调训练

当社区贡献数据积累足够后，手动触发训练：

```bash
# 通过 GitHub Actions 手动触发
# Actions → Train Model → Run workflow
```

或本地训练：
```bash
python scripts/train.py --config config/training_config.yaml
```

## 数据贡献

详见 [datasets/README.md](datasets/README.md)。

RustEpubReader 客户端会在用户同意后，通过 GitHub Device Flow 授权，自动将纠错数据以 PR 形式提交到 `datasets/submissions/` 目录。

## CDN 分发

模型文件通过 `dl.zhongbai233.com` CDN 分发，自动从 GitHub Releases 同步：

| 文件 | CDN URL |
|---|---|
| 模型 | `https://dl.zhongbai233.com/models/csc-macbert-int8.onnx` |
| 词表 | `https://dl.zhongbai233.com/models/csc-vocab.txt` |
| 清单 | `https://dl.zhongbai233.com/models/csc-manifest.json` |

## License

本仓库代码遵循 [MIT License](LICENSE)。

模型权重基于 [shibing624/macbert4csc-base-chinese](https://huggingface.co/shibing624/macbert4csc-base-chinese)，遵循其原始许可证 (Apache-2.0)。

社区贡献的数据集遵循 [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)。
