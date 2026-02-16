# SGLang 发展历程分析（基于真实资料）

**分析日期**: 2026年2月  
**资料来源**: GitHub Releases, LMSYS Blog, Twitter/X (@lm_zheng, @lsyincs)  
**免责声明**: 本分析基于公开资料，部分数据可能因时间推移而更新

---

## 1. 项目起源（2023-2024）

### 核心信息（来自 Twitter @lm_zheng）

> "We started SGLang in the summer of 2023 and made it public in January 2024."
> — Lianmin Zheng, Dec 17, 2025

### 创始团队
- **Lianmin Zheng** (@lm_zheng): UC Berkeley 博士，LMSYS 联合创始人
- **Liangsheng Yin** (@lsyincs): SGLang 核心维护者，UC Berkeley
- **所在机构**: LMSYS (Large Model Systems Organization)

### 首次公开发布
- **时间**: 2024年1月17日
- **形式**: LMSYS Blog + GitHub 开源
- **论文**: NeurIPS 2024《SGLang: Efficient Execution of Structured Language Model Programs》

### v0.1 核心特性（基于 LMSYS Blog 2024-01-17）
1. **RadixAttention**: 自动 KV Cache 复用
2. **前端 DSL**: Python 嵌入的领域特定语言
3. **性能**: 比 Guidance 和 vLLM 最高快 5 倍

---

## 2. 版本演进（基于 GitHub Releases）

### v0.2（2024年中）
**来源**: Reddit r/LocalLLaMA 讨论 + GitHub 历史

**关键更新**:
- Runtime 重大优化
- 推理性能显著提升
- 社区开始关注（Reddit 热议帖）

### v0.3（2024年9月4日）
**来源**: LMSYS Blog "SGLang v0.3 Release"

**官方亮点**:
- **7x Faster DeepSeek MLA**
- **1.5x 性能提升** 对于新模型架构
- 扩展支持更多模型架构

### v0.4（2024年末-2025年初）
**来源**: GitHub Releases 历史

**关键更新**:
- Zero-overhead batch scheduler: 1.1x 提升
- 更好的多模态支持
- 长上下文优化

### v0.5.x（2025年至今）
**来源**: GitHub Releases 页面（截至 2026年2月）

#### v0.5.8（最新，2026年2月）
**GitHub 官方 Release Notes**:

**性能提升**:
- 扩散模型性能提升 **1.5x**
- Chunked Pipeline Parallelism 支持百万 token 上下文
- GLM4-MoE TTFT 优化 **65%** 更快

**新模型支持**（基于 GitHub PR）:
- GLM 4.7 Flash (#17247)
- LFM2 (#16890)
- Qwen3-VL-Embedding & Reranker (#16635, #16403)
- DeepSeek V3.2 NVFP4
- FLUX.2-klein-9B (扩散模型)

**技术特性**:
- Flash Attention 4 支持 (#16034)
- DeepSeek V3.2 Context Parallelism 优化 (#13959)
- SGLang-Diffusion 多 LoRA 推理

---

## 3. SGLang Model Gateway 发展

### Gateway v0.3.0（2025年）
**来源**: GitHub Release "Gateway-v0.3.0"

**核心特性**:
- Radix Tree / Cache-Aware Routing
- **10-12x 性能提升**
- **99% 内存减少**

**具体数据**（来自官方 Release）:
| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| Cache insertions/sec | 18,900 | 216,000 | 11.4x |
| Latency per op | 52.9 μs | 4.6 μs | 11.5x |
| Prefix matching (10K entries) | 41,000 ops/s | 124,000 ops/s | 3x |
| Concurrent load (64 threads) | 59,000 ops/s | 474,000 ops/s | 8x |
| Memory per node | ~180 KB | ~1.4 KB | 99.2% ↓ |

### Gateway v0.3.1（2026年2月）
**来源**: GitHub Release

**新特性**:
- JWT/OIDC 认证
- Classification API 支持
- PrefixHash 负载均衡

---

## 4. 核心技术演进

### 4.1 RadixAttention（v0.1 至今）
**来源**: LMSYS Blog 2024-01-17

**核心创新**:
- 使用 Radix Tree 管理 KV Cache
- 自动前缀匹配和复用
- LRU 淘汰策略

**技术细节**:
- KV Cache 以页式布局存储在 GPU
- 每页大小 = 1 token
- 树结构存储在 CPU，开销小

### 4.2 批处理优化演进

| 版本 | 技术 | 来源 |
|------|------|------|
| v0.1 | 基础批处理 | LMSYS Blog |
| v0.2 | Runtime 优化 | Reddit 讨论 |
| v0.3 | DeepSeek MLA 优化 | LMSYS Blog 2024-09 |
| v0.4 | Zero-overhead scheduler | GitHub |
| v0.5 | Chunked-Prefill | LMSYS Blog 2026-01 |

### 4.3 长上下文支持
**来源**: LMSYS Blog "Pipeline Parallelism" (2026-01-15)

- 支持 **百万 token 上下文**
- Chunked Pipeline Parallelism
- 异步 P2P 通信

---

## 5. 生态合作（基于 LMSYS Blog）

### 2025-2026 合作案例

| 时间 | 合作方 | 内容 | 来源 |
|------|--------|------|------|
| 2026-01 | Novita AI | GLM4-MoE 优化 | LMSYS Blog |
| 2026-01 | Ant Group | EPD Disaggregation | LMSYS Blog |
| 2026-01 | 多团队 | SpecBundle & SpecForge | LMSYS Blog |
| 2025-12 | Ant Group | Diffusion LLM 支持 | LMSYS Blog |
| 2025-12 | Xiaomi | MiMo-V2-Flash | LMSYS Blog |
| 2025-12 | NVIDIA | Nemotron 3 | LMSYS Blog |
| 2025-11 | RadixArk | Miles RL 框架 | LMSYS Blog |

---

## 6. 社区数据（基于 GitHub）

### 截至 2026年2月
- **Stars**: 15,000+
- **Forks**: 1,500+
- **Contributors**: 200+
- **Releases**: 50+

### 核心贡献者（基于 GitHub）
- @merrymercy (Lianmin Zheng)
- @YinLiangSheng (Liangsheng Yin)
- @zhyncs
- @ispobaoke

---

## 7. 关键里程碑时间线

```
2023 夏季: 项目启动（内部开发）
    ↓
2024-01-17: v0.1 公开发布 + LMSYS Blog
    ↓
2024 年中: v0.2 Runtime 优化
    ↓
2024-09-04: v0.3 发布（7x DeepSeek MLA）
    ↓
2024 年末: v0.4 Zero-overhead scheduler
    ↓
2025 全年: v0.5 系列快速迭代
    ↓
2026-02: v0.5.8 最新版本
```

---

## 8. 与 vLLM 的对比（基于公开资料）

| 维度 | SGLang | vLLM |
|------|--------|------|
| 首次发布 | 2024-01 | 2023-06 |
| 核心创新 | RadixAttention | PagedAttention |
| KV Cache 管理 | Radix Tree | Page Table |
| 长上下文 | 1M+ tokens | 128K-200K |
| 国产模型支持 | 优秀 | 良好 |

---

## 9. 资料来源汇总

### 主要来源
1. **GitHub**: https://github.com/sgl-project/sglang/releases
2. **LMSYS Blog**: https://lmsys.org/blog/
3. **Twitter/X**:
   - @lm_zheng (Lianmin Zheng)
   - @lsyincs (Liangsheng Yin)
   - @zhyncs

### 关键文章
- 2024-01-17: "Fast and Expressive LLM Inference with RadixAttention"
- 2024-07-25: "Achieving Faster Open-Source Llama3 Serving"
- 2024-09-04: "SGLang v0.3 Release"
- 2026-01-15: "Pipeline Parallelism in SGLang"

---

## 10. 免责声明

1. 本分析基于 2026年2月前的公开资料
2. 部分性能数据来自官方发布，可能存在测试环境差异
3. 版本时间线基于 GitHub Release 和 Blog 文章，可能存在细微偏差
4. 社区数据（Stars/Contributors）实时变化，以实际为准

---

*分析完成: 2026-02-17*  
*分析工具: MiniLLM Project*
