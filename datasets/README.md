# 数据集贡献指南

## 数据格式

每条纠错记录为一行 JSON (JSONL 格式)：

```json
{"input": "今天天汽不错", "output": "今天天气不错"}
{"input": "我门一起去", "output": "我们一起去"}
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `input` | string | ✅ | 含错别字的原文片段，≤50 字 |
| `output` | string | ✅ | 纠正后的正确文本，≤50 字 |
| `instruction` | string | ❌ | 可选的上下文说明 |

### 约束
- `input` 和 `output` **长度必须相同**（字符级纠正，不增删字符）
- `input` ≠ `output`（需要有实际差异）
- 单个文件不超过 1 MB
- 文件扩展名必须为 `.jsonl`

## 贡献方式

### 自动贡献（推荐）

RustEpubReader 客户端在阅读修正模式下，会收集你采纳的纠错数据。当积累足够后，客户端会：

1. 征求你的同意
2. 通过 GitHub Device Flow 授权你的 GitHub 账户
3. 自动 Fork 本仓库到你的账户
4. 提交数据文件到 `datasets/submissions/`
5. 发起 Pull Request

### 手动贡献

1. Fork 本仓库
2. 在 `datasets/submissions/` 下创建 JSONL 文件，命名建议：`{github_username}_{timestamp}.jsonl`
3. 确保数据符合上述格式要求
4. 提交 PR

### 本地验证

```bash
python scripts/validate_submissions.py datasets/submissions/your_file.jsonl
```

## 数据审核

- 所有数据提交通过 PR 进行，CI 自动校验格式
- 仓库维护者会人工审核内容质量
- 恶意内容（广告、有害内容等）将被拒绝

## 隐私保护

- 数据仅包含短片段（≤50 字），不含完整文本
- 不包含任何个人身份信息
- 所有贡献数据遵循 [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) 协议
