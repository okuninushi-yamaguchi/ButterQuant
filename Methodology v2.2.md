# 📊 免费数据源全面解析

非常务实的问题！让我详细说明在**完全免费**的约束下，如何获取这些数据。

---

## 1️⃣ 核心数据源：yfinance能提供什么？

### **✅ yfinance可以获取的数据**

```python
import yfinance as yf
import pandas as pd
import numpy as np

# 基础股票数据
ticker = yf.Ticker("AAPL")

# 1. 历史价格数据（日级）
price_data = ticker.history(period="1y", interval="1d")
# 包含：Open, High, Low, Close, Volume

# 2. 期权链数据（当前快照）
expiration_dates = ticker.options  # 所有到期日列表
option_chain = ticker.option_chain('2024-01-19')  # 指定日期的期权链

# option_chain包含：
# - calls: DataFrame
#   - strike, lastPrice, bid, ask, volume, openInterest, impliedVolatility
# - puts: DataFrame（同样字段）

# 3. 无风险利率（间接）
# yfinance没有直接提供，需要从其他标的推断
treasury = yf.Ticker("^IRX")  # 13周国债利率
rf_rate = treasury.history(period="1d")['Close'].iloc[-1] / 100
```

### **❌ yfinance无法直接获取的数据**

```python
无法获取：
1. 历史期权链数据（只能获取当前快照）
2. 分钟级以下的高频数据（免费版限制）
3. 历史IV曲面
4. 历史Bid-Ask Spread
5. 逐笔tick数据
```

---

## 2️⃣ VWAP数据获取方案

### **方案A：自己计算VWAP（推荐，完全免费）**

```python
# VWAP公式：VWAP = Σ(Price × Volume) / Σ(Volume)

def calculate_vwap(ticker, date, period='1d'):
    """
    日级VWAP：用日内 Typical Price 近似
    
    Typical Price = (High + Low + Close) / 3
    """
    stock = yf.Ticker(ticker)
    
    # 获取日级数据
    df = stock.history(start=date - pd.Timedelta(days=30), 
                       end=date, 
                       interval='1d')
    
    # 计算Typical Price
    df['TypicalPrice'] = (df['High'] + df['Low'] + df['Close']) / 3
    
    # 计算VWAP
    df['VWAP'] = (df['TypicalPrice'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    return df['VWAP']

# 使用示例
vwap = calculate_vwap('AAPL', pd.Timestamp.today())
```

### **方案B：分钟级VWAP（yfinance免费支持）**

```python
def calculate_intraday_vwap(ticker, date=None):
    """
    使用yfinance的分钟级数据计算更精确的VWAP
    
    注意：免费版限制最近7天的分钟数据
    """
    stock = yf.Ticker(ticker)
    
    # 获取分钟级数据（最近7天可用）
    df = stock.history(period='7d', interval='1m')
    
    if date:
        # 筛选指定日期
        df = df[df.index.date == date]
    
    # 计算精确VWAP
    df['PV'] = df['Close'] * df['Volume']
    df['VWAP'] = df['PV'].cumsum() / df['Volume'].cumsum()
    
    # 返回当日收盘时的VWAP
    return df['VWAP'].iloc[-1]

# 使用示例
today_vwap = calculate_intraday_vwap('AAPL')
```

### **⚠️ 日级回测的VWAP妥协方案**

```python
# 对于日级回测，VWAP可以简化为：

def simple_vwap_proxy(df):
    """
    用Volume Weighted MA近似VWAP
    足够用于日级回测
    """
    window = 20  # 20天窗口
    
    df['VWAP_proxy'] = (
        (df['Close'] * df['Volume']).rolling(window).sum() / 
        df['Volume'].rolling(window).sum()
    )
    
    return df

# 实际上，对日级回测：
# VWAP ≈ 20日成交量加权移动平均
```

---

## 3️⃣ 合成市场数据的完整免费方案

### **核心思路：从有限数据中推断分布**

```python
# ==========================================
# 步骤1：获取基础数据（全部免费）
# ==========================================

import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats

class MarketDataSynthesizer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        
    def collect_baseline_stats(self):
        """
        收集真实市场统计特征（一次性）
        """
        # 1. 获取最近一个月的期权链快照
        expiration_dates = self.stock.options[:4]  # 前4个到期日
        
        spread_data = []
        volume_data = []
        oi_data = []
        
        for exp in expiration_dates:
            chain = self.stock.option_chain(exp)
            
            for option_type in ['calls', 'puts']:
                df = getattr(chain, option_type)
                
                # 计算Bid-Ask Spread %
                df['spread_pct'] = (df['ask'] - df['bid']) / ((df['ask'] + df['bid']) / 2) * 100
                
                # 按钱性分类
                current_price = self.stock.history(period='1d')['Close'].iloc[-1]
                df['moneyness'] = df['strike'] / current_price
                df['moneyness_category'] = pd.cut(
                    df['moneyness'],
                    bins=[0, 0.95, 1.05, np.inf],
                    labels=['OTM', 'ATM', 'ITM']
                )
                
                # 收集统计数据
                spread_data.append(df[['moneyness_category', 'spread_pct', 'volume', 'openInterest']])
        
        # 合并所有数据
        all_spreads = pd.concat(spread_data, ignore_index=True)
        
        # 计算分布参数
        self.spread_distributions = {}
        for category in ['OTM', 'ATM', 'ITM']:
            subset = all_spreads[all_spreads['moneyness_category'] == category]['spread_pct']
            subset = subset[subset > 0]  # 去除异常值
            
            # 拟合对数正态分布
            if len(subset) > 10:
                shape, loc, scale = stats.lognorm.fit(subset, floc=0)
                self.spread_distributions[category] = {
                    'distribution': 'lognormal',
                    'params': (shape, loc, scale),
                    'mean': subset.mean(),
                    'std': subset.std()
                }
        
        # 流动性统计
        self.liquidity_stats = {
            'volume_mean': all_spreads['volume'].mean(),
            'volume_std': all_spreads['volume'].std(),
            'oi_mean': all_spreads['openInterest'].mean(),
            'oi_std': all_spreads['openInterest'].std()
        }
        
        return self.spread_distributions, self.liquidity_stats

    # ==========================================
    # 步骤2：合成历史期权链
    # ==========================================
    
    def synthesize_option_chain(self, date, underlying_price, dte):
        """
        为历史某一天合成期权链
        
        Args:
            date: 目标日期
            underlying_price: 当天股价
            dte: 到期天数
        """
        # 1. 生成行权价网格
        strikes = self.generate_strikes(underlying_price)
        
        # 2. 获取历史波动率（用于BS定价）
        historical_vol = self.get_historical_volatility(date, window=30)
        
        # 3. 无风险利率
        rf_rate = self.get_risk_free_rate(date)
        
        # 4. 计算BS理论价格
        synthetic_chain = []
        
        for strike in strikes:
            moneyness = strike / underlying_price
            
            # 确定钱性类别
            if moneyness < 0.95:
                category = 'OTM'
            elif moneyness < 1.05:
                category = 'ATM'
            else:
                category = 'ITM'
            
            # Call期权
            call_price = self.black_scholes(
                S=underlying_price,
                K=strike,
                T=dte/365,
                r=rf_rate,
                sigma=historical_vol,
                option_type='call'
            )
            
            # 合成Bid-Ask Spread
            spread_pct = self.sample_spread(category)
            mid_price = call_price
            call_bid = mid_price * (1 - spread_pct/200)
            call_ask = mid_price * (1 + spread_pct/200)
            
            # 合成流动性
            volume = self.sample_volume(category)
            open_interest = self.sample_open_interest(category)
            
            # Put期权（用Put-Call Parity）
            put_price = call_price - underlying_price + strike * np.exp(-rf_rate * dte/365)
            put_bid = put_price * (1 - spread_pct/200)
            put_ask = put_price * (1 + spread_pct/200)
            
            synthetic_chain.append({
                'strike': strike,
                'call_bid': call_bid,
                'call_ask': call_ask,
                'call_last': mid_price,
                'call_volume': volume,
                'call_openInterest': open_interest,
                'put_bid': put_bid,
                'put_ask': put_ask,
                'put_last': put_price,
                'put_volume': volume,
                'put_openInterest': open_interest,
                'impliedVolatility': historical_vol
            })
        
        return pd.DataFrame(synthetic_chain)
    
    # ==========================================
    # 辅助函数
    # ==========================================
    
    def generate_strikes(self, price, num_strikes=21):
        """生成行权价网格"""
        # 以当前价格为中心，±20%范围，间隔5%
        strikes = []
        for i in range(-10, 11):
            strike = price * (1 + i * 0.05)
            strikes.append(round(strike / 5) * 5)  # 四舍五入到5的倍数
        return sorted(set(strikes))
    
    def get_historical_volatility(self, date, window=30):
        """计算历史波动率"""
        end_date = date
        start_date = date - pd.Timedelta(days=window+10)
        
        df = self.stock.history(start=start_date, end=end_date)
        
        # 对数收益率
        returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        
        # 年化波动率
        vol = returns.std() * np.sqrt(252)
        
        return vol
    
    def get_risk_free_rate(self, date):
        """获取无风险利率（近似）"""
        try:
            # 尝试获取国债利率
            treasury = yf.Ticker("^IRX")
            df = treasury.history(start=date - pd.Timedelta(days=5), end=date)
            if not df.empty:
                return df['Close'].iloc[-1] / 100
        except:
            pass
        
        # 默认值
        return 0.04  # 4%
    
    def sample_spread(self, category):
        """从分布中采样Spread"""
        dist_params = self.spread_distributions.get(category)
        if dist_params:
            shape, loc, scale = dist_params['params']
            spread = stats.lognorm.rvs(shape, loc, scale)
            return min(spread, 50)  # 上限50%
        else:
            # 默认值
            return {'OTM': 10, 'ATM': 5, 'ITM': 7}.get(category, 10)
    
    def sample_volume(self, category):
        """采样成交量"""
        base_volume = self.liquidity_stats['volume_mean']
        volume = np.random.lognormal(np.log(base_volume + 1), 0.5)
        return max(int(volume), 0)
    
    def sample_open_interest(self, category):
        """采样持仓量"""
        base_oi = self.liquidity_stats['oi_mean']
        oi = np.random.lognormal(np.log(base_oi + 1), 0.5)
        return max(int(oi), 0)
    
    def black_scholes(self, S, K, T, r, sigma, option_type='call'):
        """BS定价公式"""
        from scipy.stats import norm
        
        if T <= 0:
            # 到期日
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        
        return max(price, 0.01)  # 最小价格0.01
```

---

## 4️⃣ 完整使用示例

```python
# ==========================================
# 初始化并收集基准统计
# ==========================================

synthesizer = MarketDataSynthesizer('AAPL')

# 第一次运行：收集真实市场统计（保存下来重复使用）
spread_dist, liquidity_stats = synthesizer.collect_baseline_stats()

# 可以保存这些参数
import pickle
with open('aapl_market_stats.pkl', 'wb') as f:
    pickle.dump({
        'spread_distributions': spread_dist,
        'liquidity_stats': liquidity_stats
    }, f)

# ==========================================
# 回测时：为历史每一天合成期权链
# ==========================================

def backtest_with_synthetic_data(ticker, start_date, end_date):
    synthesizer = MarketDataSynthesizer(ticker)
    
    # 加载预先收集的统计参数
    with open(f'{ticker.lower()}_market_stats.pkl', 'rb') as f:
        stats = pickle.load(f)
        synthesizer.spread_distributions = stats['spread_distributions']
        synthesizer.liquidity_stats = stats['liquidity_stats']
    
    # 获取历史价格
    price_df = synthesizer.stock.history(start=start_date, end=end_date)
    
    results = []
    
    for date, row in price_df.iterrows():
        underlying_price = row['Close']
        
        # 为这一天合成30天期权链
        synthetic_chain = synthesizer.synthesize_option_chain(
            date=date,
            underlying_price=underlying_price,
            dte=30
        )
        
        # 运行你的蝴蝶策略评分
        best_butterfly = find_best_butterfly(synthetic_chain, underlying_price)
        
        # 模拟执行（考虑合成的bid-ask spread）
        execution_cost = simulate_execution(best_butterfly, synthetic_chain)
        
        results.append({
            'date': date,
            'butterfly': best_butterfly,
            'cost': execution_cost
        })
    
    return pd.DataFrame(results)

# 运行回测
results = backtest_with_synthetic_data('AAPL', '2023-01-01', '2024-01-01')
```

---

## 5️⃣ 数据质量优化技巧

### **技巧1：多标的统计平均**

```python
def collect_market_wide_stats(tickers=['AAPL', 'MSFT', 'GOOGL', 'TSLA']):
    """
    从多个标的收集统计，提高鲁棒性
    """
    all_spread_dists = []
    all_liquidity_stats = []
    
    for ticker in tickers:
        synthesizer = MarketDataSynthesizer(ticker)
        spread_dist, liq_stats = synthesizer.collect_baseline_stats()
        all_spread_dists.append(spread_dist)
        all_liquidity_stats.append(liq_stats)
    
    # 平均化参数
    averaged_spread_dist = {}
    for category in ['OTM', 'ATM', 'ITM']:
        means = [d[category]['mean'] for d in all_spread_dists if category in d]
        stds = [d[category]['std'] for d in all_spread_dists if category in d]
        
        averaged_spread_dist[category] = {
            'mean': np.mean(means),
            'std': np.mean(stds)
        }
    
    return averaged_spread_dist
```

### **技巧2：定期更新统计参数**

```python
# 每周或每月重新收集一次统计数据
# 避免市场状态变化导致模拟失真

from datetime import datetime

def refresh_stats_if_needed(ticker, stats_file, max_age_days=30):
    """
    检查统计文件是否过期，如果是则更新
    """
    if os.path.exists(stats_file):
        file_time = os.path.getmtime(stats_file)
        age_days = (datetime.now().timestamp() - file_time) / 86400
        
        if age_days < max_age_days:
            # 文件还新鲜，直接加载
            with open(stats_file, 'rb') as f:
                return pickle.load(f)
    
    # 文件过期或不存在，重新收集
    print(f"Refreshing market stats for {ticker}...")
    synthesizer = MarketDataSynthesizer(ticker)
    spread_dist, liq_stats = synthesizer.collect_baseline_stats()
    
    stats = {
        'spread_distributions': spread_dist,
        'liquidity_stats': liq_stats,
        'updated_at': datetime.now()
    }
    
    with open(stats_file, 'wb') as f:
        pickle.dump(stats, f)
    
    return stats
```

---

## 6️⃣ 免费数据源总结

| 数据类型 | 免费来源 | 获取方式 | 限制 |
|---------|---------|---------|------|
| **历史股价** | yfinance | `yf.download()` | ✅ 无限制 |
| **当前期权链** | yfinance | `ticker.option_chain()` | ✅ 无限制（但只有快照） |
| **分钟级数据** | yfinance | `interval='1m'` | ⚠️ 仅最近7天 |
| **无风险利率** | yfinance (^IRX) | `yf.Ticker("^IRX")` | ✅ 可用 |
| **VIX指数** | yfinance | `yf.Ticker("^VIX")` | ✅ 可用 |
| **VWAP** | 自己计算 | 上文公式 | ✅ 可实现 |
| **历史IV** | 从期权链反推 | BS反推算法 | ⚠️ 需要自己计算 |
| **Bid-Ask Spread** | 统计合成 | 上文方法 | ⚠️ 需要预先收集 |

---

## 🎯 最终推荐方案

### **对于日级回测，我建议：**

```python
数据获取策略：

第1步：用yfinance获取
---------------------
✅ 历史股价（Close, Volume）
✅ 当前期权链（收集统计参数）
✅ 无风险利率

第2步：自己计算
--------------
✅ VWAP（用Volume加权MA近似）
✅ 历史波动率（对数收益率标准差）
✅ 历史IV（从当前期权链反推，假设相对稳定）

第3步：统计合成
--------------
✅ Bid-Ask Spread（从真实期权链拟合分布）
✅ 流动性约束（Volume/OI分布）
✅ 完整期权链（BS + 合成microstructure）

这套方案：
- 100%免费
- 数据质量可控
- 足够支撑日级回测
- 可以合理模拟真实交易成本
```

需要我详细展开某个具体实现吗？比如IV反推算法，或者回测主循环的代码逻辑？