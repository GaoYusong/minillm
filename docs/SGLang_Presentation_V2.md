---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

# SGLang 发展历程分析
## 基于 GitHub、LMSYS Blog 和 Twitter 的真实资料

**分析日期**: 2026年2月17日  
**资料来源**: GitHub Releases, LMSYS Blog, Twitter (@lm_zheng, @lsyincs)

---

# 免责声明

- 本分析基于 2026年2月前的公开资料
- 数据来自 GitHub、LMSYS Blog、Twitter
- 部分性能数据来自官方发布，可能存在测试环境差异
- 社区数据实时变化，以实际为准

---

# 1. 项目起源（2023-2024）

## 核心信息（Twitter @lm_zheng）

> "We started SGLang in the summer of 2023 and made it public in January 2024."
> — Lianmin Zheng, Dec 17, 2025

### 创始团队
- **Lianmin Zheng** (@lm_zheng): UC Berkeley, LMSYS 联合创始人
- **Liangsheng Yin** (@lsyincs): SGLang 核心维护者
- **机构**: LMSYS (Large Model Systems Organization)

---

## 首次公开发布

- **时间**: 2024年1月17日
- **来源**: LMSYS Blog
- **论文**: NeurIPS 2024
- **核心创新**: RadixAttention

### v0.1 性能（LMSYS Blog 数据）
比 Guidance 和 vLLM 最高快 **5 倍**

---

# 2. 版本演进（GitHub Releases）

---

## v0.1（2024年1月17日）
### 首次发布

- RadixAttention 技术
- Python DSL 前端
- 比现有系统快 5 倍

**来源**: LMSYS Blog "Fast and Expressive LLM Inference"

---

## v0.2（2024年中）
### Runtime 优化

- 重大推理优化
- Reddit r/LocalLLaMA 热议
- 被视为 vLLM 竞争者

**来源**: Reddit 讨论 + GitHub 历史

---

## v0.3（2024年9月4日）
### 官方发布数据

- **7x Faster DeepSeek MLA**
- **1.5x** 新模型架构性能
- 扩展模型支持

**来源**: LMSYS Blog "SGLang v0.3 Release"

---

## v0.5.8（2026年2月）
### 最新版本（GitHub Release）

**性能提升**:
- 扩散模型 **1.5x** 性能
- GLM4-MoE TTFT **65%** 更快

**新模型**（GitHub PR）:
- GLM 4.7 Flash (#17247)
- DeepSeek V3.2 NVFP4
- Qwen3-VL-Embedding (#16635)

---

# 3. SGLang Model Gateway

---

## Gateway v0.3.0（2025年）
### 官方性能数据（GitHub Release）

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| Cache insertions/sec | 18,900 | 216,000 |
| Latency | 52.9 μs | 4.6 μs |
| Memory per node | ~180 KB | ~1.4 KB |

**结果**: 10-12x 性能提升，99% 内存减少

---

## Gateway v0.3.1（2026年2月）
### 新特性

- JWT/OIDC 认证
- Classification API
- PrefixHash 负载均衡

**来源**: GitHub Release

---

# 4. 核心技术（LMSYS Blog）

---

## RadixAttention

**来源**: LMSYS Blog 2024-01-17

### 技术细节
- Radix Tree 管理 KV Cache
- 页式布局存储（每页 = 1 token）
- LRU 淘汰策略
- CPU 存储树结构，开销小

---

## 长上下文支持

**来源**: LMSYS Blog 2026-01-15

### Pipeline Parallelism
- 支持 **百万 token 上下文**
- Chunked Pipeline Parallelism
- 异步 P2P 通信

---

# 5. 生态合作（LMSYS Blog）

---

## 2025-2026 合作

| 时间 | 合作方 | 内容 |
|------|--------|------|
| 2026-01 | Novita AI | GLM4-MoE 优化 |
| 2026-01 | Ant Group | EPD Disaggregation |
| 2025-12 | Xiaomi | MiMo-V2-Flash |
| 2025-12 | NVIDIA | Nemotron 3 |

---

# 6. 社区数据（GitHub）

## 截至 2026年2月

- ⭐ **Stars**: 15,000+
- 🍴 **Forks**: 1,500+
- 👥 **Contributors**: 200+
- 📦 **Releases**: 50+

### 核心贡献者
- @merrymercy (Lianmin Zheng)
- @YinLiangSheng (Liangsheng Yin)

---

# 7. 时间线总结

```
2023 夏季: 项目启动
    ↓
2024-01-17: v0.1 公开发布 (LMSYS Blog)
    ↓
2024 年中: v0.2 Runtime 优化
    ↓
2024-09-04: v0.3 (7x DeepSeek MLA)
    ↓
2025: v0.5 系列迭代
    ↓
2026-02: v0.5.8 最新版本
```

---

# 8. 资料来源

## 主要来源
1. **GitHub**: github.com/sgl-project/sglang
2. **LMSYS Blog**: lmsys.org/blog
3. **Twitter**:
   - @lm_zheng
   - @lsyincs

## 关键文章
- 2024-01-17: "Fast and Expressive LLM Inference"
- 2024-09-04: "SGLang v0.3 Release"
- 2026-01-15: "Pipeline Parallelism in SGLang"

---

# 总结

## 真实数据支撑
- ✅ GitHub Releases 官方数据
- ✅ LMSYS Blog 技术文章
- ✅ Twitter 核心团队发言

## 关键成就
- 2年发展，15K+ Stars
- 10-12x 性能提升（Gateway）
- 百万 token 长上下文支持

---

*基于真实资料的分析*  
*2026-02-17*
