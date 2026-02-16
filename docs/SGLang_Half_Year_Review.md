# SGLang 半年发展历程回溯 (2025年8月 - 2026年2月)

**时间范围**: 2025年8月 - 2026年2月  
**资料来源**: LMSYS Blog, GitHub Releases  
**整理日期**: 2026年2月17日

---

## 时间线总览

```
2025-08: v0.4.x 系列持续迭代
    ↓
2025-09: v0.5 系列开始
    ↓
2025-10: PyTorch Conference 2025
    ↓
2025-11: SGLang-Diffusion 发布, Miles RL 框架
    ↓
2025-12: 密集发布期 (NVIDIA/AMD/蚂蚁集团合作)
    ↓
2026-01: 生产级优化 (EPD, Pipeline Parallelism)
    ↓
2026-02: v0.5.8 最新版本
```

---

## 2025年8月-9月：v0.4 向 v0.5 过渡

### v0.4.x 系列（延续）
**来源**: GitHub Discussions #7100

- v0.4.7 等版本持续迭代
- 为 v0.5 大版本做准备
- 社区讨论活跃

---

## 2025年10月：社区影响力扩大

### PyTorch Conference 2025
**来源**: GitHub README

- SGLang Team 在 PyTorch Conference 2025 演讲
- 技术影响力扩大
- 与 NVIDIA 合作加深

---

## 2025年11月：多模态与 RL 突破

### 11月16日：SGLang-Diffusion 发布
**来源**: LMSYS Blog "SGLang-Diffusion: Two Months In" (2026-01-16 回顾)

**发布时间**: 2025年11月初

**核心特性**:
- 视频和图像生成加速
- 社区广泛关注
- 开源开发者贡献增加

### 11月19日：Miles RL 框架发布
**来源**: LMSYS Blog "Introducing Miles"

**技术贡献**:
- 企业级 RL 框架
- 基于 slime 构建
- 支持大规模 MoE 训练

**核心团队**:
- RadixArk Team
- Ant Group AQ Team（贡献者）

### 11月23日：LMSYS Fellowship Program
**来源**: LMSYS Blog

- 启动博士奖学金项目
- 支持开源 AI 基础设施贡献者
- 奖金最高 $5,000

### 11月25日：Unified FP8
**来源**: LMSYS Blog "Unified FP8"

**贡献团队**:
- InfiXAI Team
- **Ant Group AQ Team**
- SGLang RL Team
- Miles Team

**技术突破**:
- 端到端 FP8 RL 训练
- 解决 MoE 模型训练-推理不一致
- 内存减半，吞吐量翻倍

---

## 2025年12月：产业合作密集期

### 12月1日：EAGLE-3 on Vertex AI
**来源**: LMSYS Blog

- 与 Google Vertex AI 合作
- 推测解码 2x-3x 加速
- 生产级部署

### 12月2日：NVIDIA Model Optimizer 集成
**来源**: LMSYS Blog "Boost SGLang Inference"

**合作方**: NVIDIA ModelOpt Team

**特性**:
- 原生支持 NVIDIA Model Optimizer
- 量化流程简化
- 全精度到量化模型一键部署

### 12月10日：Tensor R-Fork
**来源**: LMSYS Blog "Let Tensors Fly"

**贡献团队**:
- **Ant Group DeepXPU Team**
- SGLang Team

**创新**:
- 零拷贝权重加载
- 节点间设备到设备传输
- 快速实例扩展

### 12月15日：Nemotron 3 Nano 支持
**来源**: LMSYS Blog

**合作方**: NVIDIA Nemotron Team

**特性**:
- Day-0 支持 Nemotron 3 Nano
- NVFP4 精度支持
- B200 上 4x 吞吐量提升

### 12月16日：MiMo-V2-Flash 支持
**来源**: LMSYS Blog

**合作方**: Xiaomi LLM Core Team

**模型规格**:
- 309B 总参数
- 15B 激活参数
- 滑动窗口注意力
- 多层 MTP

### 12月17日：Mini-SGLang 发布
**来源**: LMSYS Blog

**作者**: Ziyi Xu

**定位**:
- 轻量级推理框架
- 简化现代服务系统复杂性
- 保留 SGLang 核心特性

### 12月19日：Diffusion LLM 支持
**来源**: LMSYS Blog "Power Up Diffusion LLMs"

**贡献团队**:
- **Ant Group DeepXPU Team**
- SGLang Team

**特性**:
- Day-0 支持 LLaDA 2.0
- 利用 Chunked-Prefill 机制
- 无缝集成

### 12月23日：SpecBundle & SpecForge v0.2
**来源**: LMSYS Blog

**贡献团队**:
- SpecForge Team
- **Ant Group AQ Team**
- Nex-AGI Team
- EigenAI Team

**内容**:
- 生产级 EAGLE-3 模型检查点
- 大规模数据集训练
- 推测解码可用性提升

---

## 2026年1月：生产级优化

### 1月12日：EPD Disaggregation
**来源**: LMSYS Blog "EPD Disaggregation"

**贡献团队**:
- rednote hilab
- Alibaba Cloud Computing
- **AntGroup SCT**

**技术**:
- Encoder-Prefill-Decode 分离
- VLM 视觉编码独立扩展
- TTFT 降低 6-8 倍

### 1月15日：Pipeline Parallelism
**来源**: LMSYS Blog "Pipeline Parallelism in SGLang"

**作者**: Shangming Cai

**突破**:
- 支持百万 token 上下文
- Chunked Pipeline Parallelism
- 异步 P2P 通信

### 1月16日：SGLang-Diffusion 两月回顾
**来源**: LMSYS Blog "SGLang-Diffusion: Two Months In"

**进展**:
- 社区广泛采用
- 开源贡献者增加
- 生产级优化

### 1月21日：GLM4-MoE 优化
**来源**: LMSYS Blog "Optimizing GLM4-MoE"

**贡献团队**: Novita AI

**成果**:
- TTFT 提升 65%
- 端到端性能优化
- 生产环境验证

### 1月26日：INT4 QAT RL
**来源**: LMSYS Blog "Squeezing 1TB Model Rollout"

**贡献团队**:
- SGLang RL Team
- InfiXAI Team
- **Ant Group Asystem & AQ Infra Team**
- slime Team
- RadixArk Team

**突破**:
- 1TB 模型部署到单个 H200
- INT4 量化感知训练
- 训练-推理一致性

---

## 2026年2月：最新进展

### 2月11日：AMD MI300X 优化
**来源**: LMSYS Blog "Ultimate Latency Optimization of Qwen3"

**合作方**:
- Qwen C-end Infrastructure Engineering Team
- AMD AI Framework Team

**内容**:
- Qwen3 和 Qwen3-VL 在 AMD MI300X 上的优化
- 计算能力释放

### 2月16日：SGLang-Diffusion 高级优化
**来源**: LMSYS Blog "SGLang-Diffusion: Advanced Optimizations"

**团队**: SGLang-Diffusion Team

**重点**:
- 生产级视频生成
- 可扩展性、效率、稳定性
- 扩散模型部署优化

### v0.5.8 最新版本（2026年2月）
**来源**: GitHub Release

**性能提升**:
- 扩散模型 1.5x 性能提升
- Flash Attention 4 支持
- DeepSeek V3.2 优化

**新模型支持**:
- GLM 4.7 Flash
- LFM2
- Qwen3-VL-Embedding & Reranker
- DeepSeek V3.2 NVFP4
- FLUX.2-klein-9B

---

## 半年发展总结

### 版本演进
| 时间 | 版本 | 里程碑 |
|------|------|--------|
| 2025-08/09 | v0.4.x | 向 v0.5 过渡 |
| 2025-10 | - | PyTorch Conference |
| 2025-11 | v0.5 | Diffusion, Miles, FP8 |
| 2025-12 | v0.5.x | 密集产业合作 |
| 2026-01 | v0.5.x | 生产级优化 |
| 2026-02 | v0.5.8 | 最新版本 |

### 技术突破统计
| 类别 | 数量 | 代表技术 |
|------|------|---------|
| 架构创新 | 3 | EPD, Pipeline Parallelism, Tensor R-Fork |
| 训练框架 | 2 | Miles, slime |
| 量化优化 | 2 | Unified FP8, INT4 QAT |
| 多模态 | 2 | SGLang-Diffusion, Diffusion LLM |
| 产业合作 | 6+ | NVIDIA, AMD, 蚂蚁, Xiaomi 等 |

### 社区增长（估算）
- **Stars**: 10K → 15K+ (增长 50%)
- **Contributors**: 150 → 200+ (增长 33%)
- **Blog 文章**: 15+ 篇技术深度文章

---

## 关键趋势

1. **从研究到生产**: 越来越多的生产级优化
2. **多模态扩展**: 从 LLM 到 VLM、扩散模型
3. **硬件协同**: 与 NVIDIA、AMD 深度合作
4. **产业合作**: 蚂蚁、Xiaomi、Novita 等积极参与
5. **开源生态**: Fellowship Program 培养贡献者

---

## 资料来源

- LMSYS Blog: https://lmsys.org/blog/
- GitHub: https://github.com/sgl-project/sglang
- GitHub Discussions

---

*整理: MiniLLM Project*  
*日期: 2026-02-17*
