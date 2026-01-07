# 🦋 ButterQuant

**ARIMA-GARCH驱动的蝴蝶期权量化分析平台**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red.svg)](#)

> 运用**傅立叶周期分析** + **ARIMA价格预测** + **GARCH波动率建模**，智能设计Call/Put/Iron蝴蝶期权策略，最大化风险调整后收益。

---

## 🎯 项目亮点

### **为什么选择蝴蝶期权？**
- ✅ **有限风险**：最大损失 = 净成本（通常 < $5）
- ✅ **高盈亏比**：最大收益可达成本的 2-5 倍
- ✅ **方向中性**：Delta ≈ 0，不赌涨跌
- ✅ **适合盘整**：横盘市场的利润收割机

### **ButterQuant 的核心优势**
- 🔬 **科学定价**：Black-Scholes + IV Skew调整，避免理论与市场的脱节
- 📊 **多维分析**：时域（ARIMA）+ 频域（傅立叶）+ 波动率域（GARCH）
- 🎲 **Greeks管理**：自动监控Delta/Gamma/Vega/Theta，确保策略风险中性
- 🧠 **智能推荐**：多因子评分系统（0-100），只推荐高概率机会
- 📈 **实时可视化**：价格预测、波动率曲线、周期分解一目了然

---

## 🚀 快速开始

### **1. 克隆项目**
```bash
git clone https://github.com/okuninushi-yamaguchi/ButterQuant.git
cd ButterQuant
```

### **2. 启动后端（Python）**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
> 后端运行在 `http://localhost:5000`

### **3. 启动前端（React）**
```bash
cd ..  # 返回项目根目录
npm install --legacy-peer-deps
npm run dev
```
> 前端运行在 `http://localhost:3000`

### **4. 开始分析**
访问 `http://localhost:3000`：
1. **仪表盘 (Dashboard)**：查看热门股票的实时蝴蝶策略概览。点击任意卡片的**股票代码**，即可自动跳转至分析页并开始深度分析（Click-to-Analyze）。
2. **分析器 (Analyzer)**：输入任意股票代码（如 `TSLA`），获取包含价格预测、波动率分析和Greeks详解的完整报告。

### **5. 启动每日扫描 (可选)**
若要填充策略排行榜数据，需运行批量扫描脚本：
```bash
# 在 backend 目录下
python daily_scanner.py
```
> 这将扫描 Nasdaq 100 和 S&P 500 成分股，并将结果存入 SQLite 数据库。扫描过程可能需要较长时间（30分钟+）。
> 也可以通过 API `POST /api/scan` 触发后台扫描。

---

## 📊 核心功能

### **1️⃣ 傅立叶周期分析**
```
检测市场中的隐藏节奏：
✓ 机构算法拆单特征（VWAP节奏）
✓ 主导周期识别（7-180天）
✓ 趋势vs盘整分类

→ 自动选择最优DTE（到期时间）
```

### **2️⃣ ARIMA价格预测**
```
智能预测未来7-30天价格：
✓ 自动参数选择（AIC最优）
✓ 95%置信区间
✓ 价格稳定性评估

→ 确定蝴蝶中心行权价（K2）
```

### **3️⃣ GARCH波动率建模**
```
预测未来波动率 + 检测IV错误定价：
✓ 真实期权链IV（IV Skew）
✓ GARCH预测波动率
✓ IV百分位数（历史分布）

→ 发现IV高估时机（卖方优势）
```

### **4️⃣ Black-Scholes精确定价**
```
避免理论与市场脱节：
✓ 根据钱性调整波动率（IV Skew）
✓ 叠加Bid-Ask价差（3%-10%）
✓ 流动性约束（Volume/OI过滤）

→ 计算真实执行成本
```

### **5️⃣ 策略智能推荐**
```
多因子综合评分（0-100）：
✓ 价格匹配度（35%）
✓ 波动率错误定价（30%）
✓ 价格稳定性（20%）
✓ 傅立叶周期对齐（15%）

→ STRONG_BUY / BUY / NEUTRAL / AVOID
```

---

## 🏗️ 技术架构

```
┌─────────────────────────────────────────┐
│           前端 (React + Vite)           │
│  - 数据可视化 (Recharts)                │
│  - 实时图表 (价格/波动率/周期)          │
│  - 响应式设计 (Tailwind CSS)            │
└──────────────┬──────────────────────────┘
               │ HTTP API
┌──────────────▼──────────────────────────┐
│           后端 (Flask API)              │
│  - yfinance (数据获取)                  │
│  - statsmodels (ARIMA建模)              │
│  - arch (GARCH建模)                     │
│  - numpy/scipy (傅立叶分析)             │
│  - Black-Scholes定价引擎                │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│        Yahoo Finance API                │
│  - 历史价格 (2年数据)                   │
│  - 期权链 (Strike/IV/Volume/OI)         │
│  - 无风险利率 (^IRX)                    │
└─────────────────────────────────────────┘
```

---

## 🎓 核心方法论

### **傅立叶分析的正确姿势**
❌ **错误做法**：直接对价格序列做FFT
```python
fft(prices)  # 会产生虚假低频能量！
```

✅ **正确做法**：去趋势后再分析
```python
# 方法1: 相对VWAP的偏移（检测机构行为）
detrended = prices - VWAP(prices, volume)

# 方法2: 对数收益率（标准金融方法）
returns = log(prices[t] / prices[t-1])

fft(detrended)  # 检测真实周期
```

### **IV Skew的重要性**
同一个蝴蝶策略，如果忽略IV Skew，**定价误差高达20%**！

```
示例（现价 $200）：

不考虑Skew（错误）：
  $180 Call @ σ=25% → $21.50
  $200 Call @ σ=25% → $8.00
  $220 Call @ σ=25% → $4.50
  净成本 = $1.00

考虑Skew（正确）：
  $180 Call @ σ=26% → $22.30  ← ITM区域IV略高
  $200 Call @ σ=25% → $8.00   ← ATM基准
  $220 Call @ σ=23% → $3.80   ← OTM Call IV低
  净成本 = $1.20
  
误差 = 20%！
```

### **Greeks风险管理**
理想的蝴蝶策略特征：
- **Delta ≈ 0**：方向中性（|Δ| < 0.10）
- **Gamma > 0**：接近中心时加速获利
- **Vega < 0**：做空波动率（IV高位入场）
- **Theta > 0**：时间是朋友（每天+$0.05~$0.15）

---

## 📈 策略类型

### **Call Butterfly（看涨盘整）**
```
上翼 $220 Call → $0.50 (买入)
中心 $200 Call → $8.20 × 2 (卖出)
下翼 $180 Call → $21.50 (买入)

净成本: $5.60
最大收益: $14.40 (价格到$200时)
盈亏比: 2.57:1
```

**适用场景**：
- ✅ 傅立叶：上涨趋势 + 周期底部
- ✅ ARIMA：预测价格在中心附近
- ✅ GARCH：IV高位（>60百分位）

---

### **Put Butterfly（看跌盘整）**
```
上翼 $220 Put → $22.80 (买入)
中心 $200 Put → $9.10 × 2 (卖出)
下翼 $180 Put → $1.20 (买入)

净成本: $5.80
最大收益: $14.20 (价格到$200时)
盈亏比: 2.45:1
```

**适用场景**：
- ✅ 傅立叶：下跌趋势 + 周期顶部
- ✅ ARIMA：预测价格在中心附近
- ✅ GARCH：IV高位

---

### **Iron Butterfly（中性盘整）**
```
上翼 $220 Call → $0.50 (买入)
中心 $200 Put → $9.10 (卖出)
中心 $200 Call → $8.20 (卖出)
下翼 $180 Put → $1.20 (买入)

净收入: $15.60 ✅ (卖期权为主)
最大收益: $15.60
最大损失: $4.40 (价格远离$200)
盈亏比: 3.55:1
```

**适用场景**：
- ✅ 傅立叶：横盘趋势
- ✅ ARIMA：价格稳定性高（CI < 8%）
- ✅ GARCH：IV极高位（>75百分位）

---

## 🔬 回测框架（规划中）

由于 `yfinance` 只提供当日期权链，我们采用**合成市场数据**方案：

```python
# 步骤1: 收集真实市场统计
bid_ask_spreads = collect_real_spreads(days=30)
# OTM: mean=10%, std=3%
# ATM: mean=5%, std=2%

# 步骤2: 回测时合成期权链
for 历史每一天:
    理论价格 = black_scholes(S, K, T, r, σ_historical)
    实际价格 = 理论价格 × (1 + sample_spread())
    
    if liquidity_ok and score > 70:
        execute_trade()
```

---

## 🛠️ 技术栈

### **后端**
- **Flask 3.0** - RESTful API
- **yfinance** - Yahoo Finance数据
- **statsmodels** - ARIMA时间序列
- **arch** - GARCH波动率建模
- **numpy/scipy** - 傅立叶变换
- **pandas** - 数据处理

### **前端**
- **React 18** - UI框架
- **Vite** - 构建工具
- **Recharts** - 图表库
- **Tailwind CSS** - 样式框架
- **Lucide React** - 图标库
- **i18next** - 国际化支持
- **Standardized Structure** - `src/` 目录架构

---

## 📁 项目结构

```
ButterQuant/
├── backend/
│   ├── app.py                 # Flask主应用
│   ├── requirements.txt       # Python依赖
│   └── [分析算法模块]
├── src/
│   ├── assets/                # 静态资源 (Logo等)
│   ├── components/            # React组件
│   ├── locales/               # 国际化语言包
│   ├── App.tsx                # 主应用组件
│   ├── main.tsx               # 入口文件
│   └── config.ts              # 配置文件
├── public/                    # 静态资源
├── docs/
│   ├── Methodology v3.0.md    # 方法论详解
│   └── README.md              # 运行指南
├── package.json
├── vite.config.ts
└── README.md                  # 本文件
```

---

## 📖 文档

- **[完整方法论](docs/Methodology%20v3.0.md)** - 数学模型详解
- **[运行指南](docs/README.md)** - 环境配置与故障排除
- **[API文档]** - 接口说明（规划中）

---

## 🎯 路线图

### **已完成 ✅**
- [x] 傅立叶周期分析（VWAP去趋势）
- [x] ARIMA价格预测（自动选参）
- [x] GARCH波动率建模
- [x] Black-Scholes定价（IV Skew）
- [x] Greeks计算（Delta/Gamma/Vega/Theta）
- [x] 多因子评分系统
- [x] Call/Put/Iron蝴蝶策略
- [x] 实时可视化（价格/波动率/周期）
- [x] 现代化UI重构 (Tab布局/响应式设计的仪表盘/Logo集成)
- [x] 一键分析导航 (Click-to-Analyze)

### **进行中 🚧**
- [ ] 回测框架（合成市场数据）
- [ ] 流动性过滤（Volume/OI）
- [ ] 滑点建模（三因子分解）
- [ ] 止损止盈逻辑

### **规划中 🔮**
- [ ] 机器学习增强（XGBoost）
- [ ] 多策略组合优化
- [ ] 实时监控与预警
- [ ] 移动端适配

---

## 💬 反馈与建议

目前本项目**暂不接受代码贡献**，但欢迎：

- 🐛 **提交Bug报告** - 通过Issue描述问题
- 💡 **功能建议** - 分享你的想法和需求
- 📖 **文档改进** - 指出文档中的错误或不清楚的地方
- ⭐ **Star支持** - 如果觉得有用请给个Star

### 如何提交Issue：
1. 点击 **Issues** 标签
2. 选择 **New Issue**
3. 清楚描述问题或建议
4. 附上截图或错误日志（如适用）

---

## ⚠️ 免责声明

**重要提示：请仔细阅读**

本项目仅供**学习、研究和展示**使用，不构成任何形式的投资建议、推荐或指导。期权交易具有**极高风险**，可能导致**本金全部损失**，甚至超额亏损。

### 风险警告

1. ⚠️ **本工具的分析结果不保证准确性**
   - 所有预测基于历史数据和统计模型
   - 过去表现不代表未来结果
   - 模型存在误差和局限性

2. ⚠️ **期权交易风险极高**
   - 可能在短时间内损失全部投资
   - 某些策略（如卖方策略）可能面临无限损失
   - 流动性风险可能导致无法平仓

3. ⚠️ **使用本工具前，您必须：**
   - ✅ 充分了解期权交易规则和风险
   - ✅ 具备相应的风险承受能力
   - ✅ 在模拟账户中充分测试
   - ✅ 咨询持牌专业金融顾问
   - ✅ 遵守当地法律法规

4. ⚠️ **责任限制**
   - 作者不对使用本工具导致的任何盈利或亏损负责
   - 用户自行承担所有交易风险和后果
   - 本工具按"现状"提供，不提供任何明示或暗示的保证

### 适用法律

本项目及其使用受您所在司法管辖区的法律约束。如果您所在地区禁止或限制期权交易，请勿使用本工具。

**请务必审慎决策，理性投资！**

---

## 📝 版权声明

**© 2025 ButterQuant. All Rights Reserved.**

本项目目前**不开放源代码许可**。这意味着：

❌ **未经许可，您不得：**
- 复制、修改或分发本代码
- 将本代码用于商业用途
- 创建基于本代码的衍生作品

✅ **您可以：**
- 查看本项目用于学习和研究
- Fork本仓库用于个人学习（不得公开分发）
- 通过Issue提出问题和建议

📧 **商业合作或授权咨询：** [mingsely@gmail.com]

> 未来可能会考虑开放源代码许可，敬请关注。

---

## 🌟 致谢

- [Yahoo Finance](https://finance.yahoo.com/) - 提供免费金融数据
- [statsmodels](https://www.statsmodels.org/) - 时间序列建模
- [arch](https://arch.readthedocs.io/) - GARCH模型
- [Recharts](https://recharts.org/) - 美观的图表组件

---

<p align="center">
  <strong>© 2025 ButterQuant. All Rights Reserved.</strong>
</p>

<p align="center">
  本项目受版权保护，未经授权不得复制、修改或分发。
</p>

<p align="center">
  <a href="#-butterquant">回到顶部</a>
</p>