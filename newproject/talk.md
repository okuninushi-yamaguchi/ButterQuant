现在有两个新增markdown文档，nas100.md和sp500.md，
开设第三个页面，件名{strategyrank.tsx}
同样是中英双语，
建立一个后台储存和自动计算机制，把nas100.md和sp500.md里列表的{ticker}，一个一个输入，计算，得出结果，因为计算过程需要时间，所以需要一个自动计算的机制，nas100（100个股票，计算100次），sp500(500个股票，计算500次)，看能否建立一个优化机制。
strategyrank.tsx页面的功能是，把optionanalyzer.tsx页面里输出的结果{"score": "智能评分"}，进行从高到底排名
有列表切换按钮，分别是{Nasdaq 100 index, S&P 500 index}（对应中文，纳斯达克100指数，标普500指数）
只显示前20名。显示方式{rank: 排名，ticker: 代码，Call/Put/Iron Butterfly策略: Call/Put/Iron Butterfly策略，score: 智能评分}，下面居中有{显示更多}按钮，点击后显示后20名，再点击后20名，显示后40名，以此类推。

{
    # ========== 基础信息 ==========
    'ticker': 'AAPL',
    'analysis_date': '2025-01-05 14:30:00',
    'current_price': 185.23,
    'forecast_price': 187.45,
    'upper_bound': 192.30,
    'lower_bound': 182.60,
    'price_stability': 5.2,  # 预测区间宽度%
    
    # ========== 傅立叶分析结果 ==========
    'fourier': {
        'trend_direction': 'UP',           # UP/DOWN/FLAT
        'trend_slope': 0.25,               # 归一化斜率%
        'cycle_position': 'TROUGH',        # PEAK/TROUGH
        'dominant_periods': [
            {
                'period': 28.3,            # 天数
                'power': 1250.5,
                'power_pct': 15.2          # 能量占比%
            },
            # ... 最多5个周期
        ],
        'dominant_period_days': 28.3,
        'period_strength': 15.2,
        'butterfly_type': 'CALL',          # CALL/PUT/IRON
        'strategy_reason': '低频上涨趋势 + 中频周期底部 → 预期上涨后盘整',
        'low_freq_signal': [185.2, 185.5, ...],   # 趋势分量（数组）
        'mid_freq_signal': [186.1, 185.8, ...]    # 周期分量（数组）
    },
    
    # ========== ARIMA预测结果 ==========
    'arima': {
        'forecast': [187.2, 187.8, 188.1, ...],   # 30天预测
        'upper_bound': [190.5, 191.2, ...],
        'lower_bound': [183.9, 184.4, ...],
        'mean_forecast': 187.45,
        'forecast_7d': 188.20,
        'forecast_30d': 189.50,
        'model_order': (2, 1, 2),         # ARIMA(p,d,q)
        'aic': 1523.4,
        'confidence_width': 6.8           # 平均置信区间宽度
    },
    
    # ========== GARCH波动率预测 ==========
    'garch': {
        'predicted_vol': 0.28,            # 年化波动率
        'current_iv': 0.32,               # 市场隐含波动率
        'historical_vol': 0.27,           # 历史波动率
        'iv_skew': {
            'atm': 0.32,                  # ATM IV
            'otm_call': 0.30,             # OTM Call IV (105%)
            'otm_put': 0.35,              # OTM Put IV (95%)
            'skew_call': -6.25,           # Call偏斜%
            'skew_put': 9.38              # Put偏斜%
        },
        'vol_mispricing': 14.3,           # IV溢价% (正=高估)
        'iv_percentile': 72.5,            # IV历史分位数
        'forecast_vol': [0.28, 0.285, ...],  # 30天预测
        'garch_params': {
            'omega': 0.00012,
            'alpha': 0.08,
            'beta': 0.90
        }
    },
    
    # ========== 蝴蝶策略设计 ==========
    'butterfly': {
        'center_strike': 270.0,
        'lower_strike': 230.0,
        'upper_strike': 310.0,
        'wing_width': 40.0,
        'dte': 30,                        # 到期天数
        
        # 各腿成本
        'lower_cost': 0.21,               # 买入Put成本
        'center_credit': 18.95,           # 卖出Straddle收入（Call+Put各$18.95）
        'upper_cost': 0.61,               # 买入Call成本
        'net_debit': 37.08,               # 净收入（Iron Butterfly是收钱）
        
        # 盈亏指标
        'max_profit': 37.08,              # 最大收益
        'max_loss': 2.92,                 # 最大损失（$40翼宽 - $37.08收入）
        'profit_ratio': 12.7,             # 盈亏比
        'breakeven_lower': 232.92,
        'breakeven_upper': 307.08,
        'prob_profit': 68.0,              # 成功概率%
        
        'risk_free_rate': 4.35,           # 无风险利率%
        
        # Greeks
        'greeks': {
            'delta': 0.02,                # 方向性
            'gamma': -0.008,              # Gamma（中心附近）
            'vega': -1.25,                # 波动率敏感度
            'theta': 0.85                 # 时间价值衰减/天
        },
        
        # 价差统计
        'spreads': {
            'lower': 10.0,                # 下翼Bid-Ask价差%
            'center': 5.0,                # 中心价差%
            'upper': 8.0                  # 上翼价差%
        }
    },
    
    # ========== 交易信号 ==========
    'signals': {
        'price_stability': True,          # 价格稳定
        'vol_mispricing': True,           # 波动率错误定价
        'trend_clear': True,              # 趋势明确
        'cycle_aligned': True,            # 周期对齐
        'delta_neutral': True,            # Delta中性
        'iv_high': True                   # IV在高位
    },
    
    # ========== 风险评估 ==========
    'risk_level': 'LOW',                  # LOW/MEDIUM/HIGH
    
    # ========== 综合评分 ==========
    'score': {
        'total': 82.3,                    # 总分0-100
        'components': {
            'price_match': 88.5,          # 价格匹配度
            'vol_mispricing': 78.2,       # 波动率错误定价
            'stability': 92.0,            # 稳定性
            'fourier_align': 100.0,       # 傅立叶对齐
            'delta_penalty': 1.0          # Delta惩罚
        },
        'recommendation': 'STRONG_BUY',   # STRONG_BUY/BUY/NEUTRAL/AVOID
        'confidence_level': 'HIGH'        # HIGH/MEDIUM/LOW
    },
    
    # ========== 交易建议 ==========
    'trade_suggestion': {
        'action': 'STRONG_BUY',
        'position_size': 'STANDARD',      # SMALL/MEDIUM/STANDARD
        'entry_timing': 'IMMEDIATE',      # IMMEDIATE/WAIT_FOR_PULLBACK
        'stop_loss': 4.38,                # 止损价格
        'take_profit': 25.96,             # 止盈价格
        'hold_until': '30 days or 70% max profit',
        'key_risks': [
            'IV不在高位，卖期权优势不明显',
            # ... 其他风险提示
        ]
    },
    
    # ========== 图表数据 ==========
    'chart_data': {
        # 傅立叶分解图（最近120天）
        'fourier': [
            {'date': '09/15', 'actual': 175.2, 'lowFreq': 174.8, 'midFreq': 175.5},
            # ...
        ],
        
        # 价格预测图（历史60天 + 未来30天）
        'price_forecast': [
            {'date': '11/25', 'actual': 182.3, 'forecast': None, 'upper': None, 'lower': None},
            {'date': '01/06', 'actual': None, 'forecast': 187.2, 'upper': 190.5, 'lower': 183.9},
            # ...
        ],
        
        # 波动率对比图（历史30天 + 未来30天）
        'volatility': [
            {'date': '12/05', 'realized': 0.27, 'predicted': None},
            {'date': '01/06', 'realized': None, 'predicted': 0.28},
            # ...
        ],
        
        # 功率谱图（前5个主周期）
        'spectrum': [
            {'period': '28天', 'power': 1250.5, 'powerPct': 15.2, 'periodDays': 28.3},
            {'period': '14天', 'power': 890.2, 'powerPct': 10.8, 'periodDays': 14.1},
            # ...
        ]
    }
}