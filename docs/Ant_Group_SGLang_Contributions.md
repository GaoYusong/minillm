# 蚂蚁集团对 SGLang 社区的贡献总结

**资料来源**: LMSYS Blog, GitHub, Twitter/X  
**时间范围**: 2025-2026年  
**整理日期**: 2026年2月17日

---

## 1. EPD Disaggregation（编码器-预填充-解码分离）

### 贡献团队
- **Ant Group**（主要贡献者）
- **rednote hilab**
- **Alibaba Cloud Computing**

### 技术贡献
**来源**: LMSYS Blog "EPD Disaggregation" (2026-01-12)

EPD (Encoder-Prefill-Decode) Disaggregation 是一种针对视觉-语言模型(VLM)的新型架构：

#### 核心创新
1. **独立扩展视觉编码能力**
   - 编码器服务器可水平扩展，不影响语言模型部署
   - 适用于视觉计算密集型工作负载

2. **与现有 PD 分离兼容**
   - 可与 Prefill-Decode 分离结合，形成完整的三层架构

3. **灵活的传输后端**
   - 支持 ZMQ、GPU-direct (via Mooncake) 等多种传输机制

4. **视觉嵌入缓存**
   - 频繁使用的图像可在编码器服务器缓存
   - 避免重复的 ViT 计算

#### 性能数据
- 在图像密集型场景（如多图像输入）中：
  - TTFT（首token时间）降低 **6-8 倍**
  - 在 1 QPS 负载下，相比同地部署显著减少延迟

#### 技术细节
- **GitHub Issue**: #8223 (Roadmap)
- **状态**: 已合并到 SGLang 主分支
- **内部版本**: AntGroup 内部版本已显示性能改进

---

## 2. slime / Miles RL 框架

### 贡献团队
- **Ant Group AQ Team**
- **slime Team**
- **RadixArk Team**

### 技术贡献
**来源**: LMSYS Blog "Introducing Miles" (2025-11-19)

#### slime 框架
- **定位**: 轻量级 RL 框架
- **应用**: 支持 GLM-4.6 等大型 MoE 模型的后训练
- **设计原则**:
  - 开放使用性能
  - 模块化设计
  - 为研究人员设计
  - 社区驱动

#### Miles 框架（基于 slime）
- **定位**: 企业级 RL 框架，支持大规模 MoE 训练和生产工作负载
- **与 SGLang 集成**: 原生支持 SGLang 进行高吞吐量推理

#### 关键特性（已上游到 slime）
1. **真正的 On-Policy 训练**
   - 通过基础设施方法支持真正的 on-policy
   - 使用 Flash Attention 3, DeepGEMM, batch invariant kernels
   - 训练和推理之间的 KL 散度为 0

2. **内存优化**
   - 错误传播避免良性 OOM 崩溃
   - 内存边距修复 NCCL 相关 OOM
   - FSDP 内存使用优化

3. **在线草稿模型训练的推测解码**
   - 在 RL 过程中对草稿模型进行在线 SFT
   - 相比冻结 MTP，rollout 速度提升 25%+
   - 支持 MTP + sequence packing + CP

---

## 3. Unified FP8（统一 FP8 精度）

### 贡献团队
- **Ant Group AQ Team**（主要贡献者）
- **InfiXAI Team**
- **SGLang RL Team**
- **Miles Team**

### 技术贡献
**来源**: LMSYS Blog "Unified FP8" (2025-11-25)

#### 核心创新
实现了**端到端 FP8 管道**用于 RL 训练和采样：

#### 解决的问题
- MoE 模型在使用 BF16 训练 + FP8 rollout 时，训练-推理差异严重
- 模型越大，差异越明显

#### 解决方案
- 训练和 rollout 统一使用 FP8
- 有效消除量化误差导致的训练-推理不一致
- 提高 RL 训练的速度和稳定性

#### 技术细节
- 支持 Qwen3-4B 和 Qwen3-30B-A3B 的 FP8 RL 训练
- 已在 miles 中完全支持，开箱即用
- 选择 FP32 作为训练时的缩放精度

#### 性能优势
1. **内存占用显著减少**: 相比 BF16，FP8 可将模型权重和激活内存减半
2. **理论计算吞吐量 2 倍**: H100 上 FP8 Tensor Cores 提供 1979 TFLOPS，是 BF16 (989 TFLOPS) 的两倍
3. **缓解内存带宽瓶颈**: 更紧凑的数据表示，减少数据传输时间

---

## 4. INT4 QAT RL（INT4 量化感知训练）

### 贡献团队
- **Ant Group Asystem & AQ Infra Team**
- **SGLang RL Team**
- **InfiXAI Team**
- **slime Team**
- **RadixArk Team**

### 技术贡献
**来源**: LMSYS Blog "Squeezing 1TB Model Rollout into a Single H200" (2026-01-26)

#### 核心创新
成功落地 **INT4 量化感知训练 (QAT) 管道**：

#### 技术方案
- 训练时使用假量化 (fake quantization)
- 推理时使用真实量化 (W4A16)
- 实现了与 BF16 全精度相当的稳定性和训练-推理一致性

#### 应用成果
- 将 1TB 模型部署到单个 H200 GPU
- 显著降低推理成本
- 保持模型性能

---

## 5. Tensor R-Fork（张量远程分叉）

### 贡献团队
- **Ant Group DeepXPU Team**
- **SGLang Team**

### 技术贡献
**来源**: LMSYS Blog "Let Tensors Fly" (2025-12-10)

#### 核心创新
一种新颖的权重加载方法：

#### 技术特点
- 利用高效的节点间设备到设备互连
- 从运行的 SGLang 实例加载张量到新实例
- **零拷贝**传输

#### 三大优势
1. **零拷贝传输**: 避免数据复制开销
2. **高效互连**: 利用 RDMA 等高速网络
3. **快速扩展**: 快速启动新实例

---

## 6. Diffusion LLM 支持

### 贡献团队
- **Ant Group DeepXPU Team**
- **SGLang Team**

### 技术贡献
**来源**: LMSYS Blog "Power Up Diffusion LLMs" (2025-12-19)

#### 核心创新
在 SGLang 中引入扩散大语言模型 (dLLM) 框架：

#### 技术实现
- 利用现有的 Chunked-Prefill 机制
- 无需核心架构变更即可集成到 SGLang 生态系统

#### 支持特性
- Day-0 支持 LLaDA 2.0
- 无缝集成

---

## 7. SpecBundle & SpecForge

### 贡献团队
- **Ant Group AQ Team**（主要贡献者）
- **SpecForge Team**
- **Nex-AGI Team**
- **EigenAI Team**

### 技术贡献
**来源**: LMSYS Blog "SpecBundle & SpecForge v0.2" (2025-12-23)

#### 核心贡献
- **SpecBundle Phase 1**: 生产级 EAGLE-3 模型检查点集合
- 在大规模数据集上训练
- 旨在提高推测解码模型的可用性

#### 性能提升
- 提高推测解码的接受长度和加速比
- 支持在线草稿模型训练

---

## 8. 其他贡献

### 持续的技术合作
- **SGLang 核心开发**: Ant Group AQ Team 成员参与 SGLang 核心代码贡献
- **性能优化**: 针对生产环境的大规模部署优化
- **Bug 修复和稳定性改进**: 通过实际生产环境反馈改进框架

---

## 贡献总结

### 按类别统计

| 类别 | 贡献数量 | 主要项目 |
|------|---------|---------|
| **架构创新** | 2 | EPD Disaggregation, Tensor R-Fork |
| **训练框架** | 2 | slime, Miles |
| **量化优化** | 2 | Unified FP8, INT4 QAT RL |
| **模型支持** | 2 | Diffusion LLM, SpecBundle |

### 技术影响力

1. **生产级可靠性**: Ant Group 的贡献侧重于生产环境的稳定性和可扩展性
2. **大规模训练**: 专注于 MoE 模型和 RL 训练框架
3. **硬件协同设计**: 充分利用 NVIDIA 新硬件特性（FP8, NVFP4）
4. **开源协作**: 所有主要贡献都通过 LMSYS Blog 和 GitHub 开源分享

### 合作模式

- **深度集成**: Ant Group 团队与 SGLang 核心团队紧密合作
- **上游优先**: 改进优先上游到 slime 和 SGLang 主分支
- **生产验证**: 所有技术先在 Ant Group 生产环境验证，再开源

---

## 参考资料

1. LMSYS Blog: "EPD Disaggregation" (2026-01-12)
2. LMSYS Blog: "Introducing Miles" (2025-11-19)
3. LMSYS Blog: "Unified FP8" (2025-11-25)
4. LMSYS Blog: "Squeezing 1TB Model Rollout" (2026-01-26)
5. LMSYS Blog: "Let Tensors Fly" (2025-12-10)
6. LMSYS Blog: "Power Up Diffusion LLMs" (2025-12-19)
7. LMSYS Blog: "SpecBundle & SpecForge v0.2" (2025-12-23)
8. GitHub: sgl-project/sglang

---

*整理: MiniLLM Project*  
*日期: 2026-02-17*
