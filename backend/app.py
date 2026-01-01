# -*- coding: utf-8 -*-
"""
ARIMA-GARCH 蝴蝶期权分析后端 API (改进版)
依赖安装: pip install flask flask-cors yfinance numpy pandas scipy statsmodels arch
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from scipy import signal
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

class ButterflyAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = None
        self.prices = None
        self.volumes = None
        
    def fetch_data(self, period='2y'):
        """获取股票数据"""
        stock = yf.Ticker(self.ticker)
        self.data = stock.history(period=period)
        
        if self.data.empty:
            raise ValueError(f"无法获取 {self.ticker} 的数据")
        
        self.prices = self.data['Close'].values
        self.volumes = self.data['Volume'].values
        return self.data
    
    def calculate_vwap(self, window=20):
        """计算成交量加权平均价格（VWAP）"""
        df = pd.DataFrame({
            'price': self.prices,
            'volume': self.volumes
        })
        
        # 计算价格*成交量
        df['pv'] = df['price'] * df['volume']
        
        # 滚动窗口VWAP
        cumsum_pv = df['pv'].rolling(window=window).sum()
        cumsum_v = df['volume'].rolling(window=window).sum()
        
        vwap = cumsum_pv / cumsum_v
        
        # 填充NaN（用后面的值向前填充）
        vwap = vwap.fillna(method='bfill')
        
        return vwap.values
    
    def fourier_analysis(self):
        """改进的傅立叶变换分析（真正的去趋势）"""
        n = len(self.prices)
        
        # 方法1：相对VWAP的偏移（去趋势）
        vwap = self.calculate_vwap(window=min(20, n // 3))
        detrended = self.prices - vwap
        
        # 去除NaN
        detrended = detrended[~np.isnan(detrended)]
        n_clean = len(detrended)
        
        if n_clean < 50:
            raise ValueError("数据量不足以进行傅立叶分析")
        
        # 加窗函数（减少频谱泄漏）
        window_func = np.hanning(n_clean)
        detrended_windowed = detrended * window_func
        
        # FFT
        fft_result = np.fft.fft(detrended_windowed)
        power_spectrum = np.abs(fft_result) ** 2
        frequencies = np.fft.fftfreq(n_clean)
        
        # 只保留正频率部分
        positive_freq_idx = frequencies > 0
        positive_freqs = frequencies[positive_freq_idx]
        positive_power = power_spectrum[positive_freq_idx]
        
        # 找主要周期（排除极端值）
        sorted_idx = np.argsort(positive_power)[::-1]
        top_periods = []
        
        for idx in sorted_idx[:10]:
            if positive_freqs[idx] > 0:
                period = 1 / positive_freqs[idx]
                # 只关注7天到180天的周期（期权相关范围）
                if 7 < period < 180:
                    top_periods.append({
                        'period': float(period),
                        'power': float(positive_power[idx]),
                        'power_pct': float(positive_power[idx] / positive_power.sum() * 100)
                    })
                    
                    if len(top_periods) >= 5:
                        break
        
        # 低频滤波（提取趋势）- 60天以上的周期
        cutoff_low = max(1, int(n_clean / 60))
        fft_filtered_low = fft_result.copy()
        fft_filtered_low[cutoff_low:-cutoff_low] = 0
        low_freq_signal = np.fft.ifft(fft_filtered_low).real
        
        # 恢复到原始价格尺度
        low_freq_signal = low_freq_signal + vwap[~np.isnan(vwap)][:len(low_freq_signal)]
        
        # 中频滤波（提取周期）- 5天到60天
        fft_filtered_mid = np.zeros_like(fft_result)
        mid_low = max(1, int(n_clean / 60))
        mid_high = min(n_clean // 2, int(n_clean / 5))
        
        fft_filtered_mid[mid_low:mid_high] = fft_result[mid_low:mid_high]
        fft_filtered_mid[-mid_high:-mid_low] = fft_result[-mid_high:-mid_low]
        mid_freq_signal = np.fft.ifft(fft_filtered_mid).real
        
        # 补齐长度
        if len(low_freq_signal) < n:
            low_freq_signal = np.pad(low_freq_signal, (n - len(low_freq_signal), 0), 
                                      mode='edge')
        if len(mid_freq_signal) < n:
            mid_freq_signal = np.pad(mid_freq_signal, (n - len(mid_freq_signal), 0), 
                                      mode='constant', constant_values=0)
        
        # 趋势判断（看最近20天的低频信号斜率）
        recent_low_freq = low_freq_signal[-20:]
        trend_slope = (recent_low_freq[-1] - recent_low_freq[0]) / len(recent_low_freq)
        
        # 归一化斜率（相对于价格）
        normalized_slope = trend_slope / self.prices[-1] * 100
        
        if normalized_slope > 0.15:
            trend_direction = 'UP'
        elif normalized_slope < -0.15:
            trend_direction = 'DOWN'
        else:
            trend_direction = 'FLAT'
        
        # 周期位置（看中频信号最近的值）
        recent_mid_freq = mid_freq_signal[-10:]
        cycle_position = 'PEAK' if np.mean(recent_mid_freq) > 0 else 'TROUGH'
        
        # 决定策略类型
        if trend_direction == 'UP' and cycle_position == 'TROUGH':
            butterfly_type = 'CALL'
            strategy_reason = '低频上涨趋势 + 中频周期底部 → 预期上涨后盘整'
        elif trend_direction == 'DOWN' and cycle_position == 'PEAK':
            butterfly_type = 'PUT'
            strategy_reason = '低频下跌趋势 + 中频周期顶部 → 预期下跌后盘整'
        elif trend_direction == 'FLAT':
            butterfly_type = 'IRON'
            strategy_reason = '低频平稳 + 无明显方向 → 铁蝴蝶（双向盘整）'
        else:
            butterfly_type = 'CALL'
            strategy_reason = f'{trend_direction}趋势 + {cycle_position}位置 → 谨慎操作'
        
        # 周期强度评估
        if top_periods:
            dominant_period = top_periods[0]['period']
            period_strength = top_periods[0]['power_pct']
        else:
            dominant_period = 30  # 默认值
            period_strength = 0
        
        return {
            'trend_direction': trend_direction,
            'trend_slope': float(normalized_slope),
            'cycle_position': cycle_position,
            'dominant_periods': top_periods,
            'dominant_period_days': float(dominant_period),
            'period_strength': float(period_strength),
            'butterfly_type': butterfly_type,
            'strategy_reason': strategy_reason,
            'low_freq_signal': low_freq_signal.tolist(),
            'mid_freq_signal': (mid_freq_signal * 3 + self.prices[-len(mid_freq_signal):]).tolist()
        }
    
    def arima_forecast(self, steps=30):
        """改进的ARIMA价格预测（自动选择最优参数）"""
        try:
            # 使用更长的训练数据（120天）
            train_length = min(120, len(self.prices))
            train_data = self.prices[-train_length:]
            
            # 自动选择最优参数
            best_aic = np.inf
            best_order = (2, 1, 2)
            best_model = None
            
            # 候选参数组合
            candidate_orders = [
                (1, 1, 1),  # 最简单
                (2, 1, 2),  # 平衡
                (1, 1, 2),  # 常用
                (2, 1, 1),  # 常用
                (3, 1, 2),  # 复杂一点
            ]
            
            for order in candidate_orders:
                try:
                    model = ARIMA(train_data, order=order)
                    fitted = model.fit()
                    
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = order
                        best_model = fitted
                except:
                    continue
            
            # 如果所有模型都失败，使用简单模型
            if best_model is None:
                model = ARIMA(train_data, order=(1, 1, 1))
                best_model = model.fit()
                best_order = (1, 1, 1)
            
            # 预测（使用更准确的置信区间）
            forecast_result = best_model.get_forecast(steps=steps)
            forecast_df = forecast_result.summary_frame(alpha=0.05)  # 95% CI
            
            # 提取预测值和置信区间
            forecast_values = forecast_df['mean'].values
            upper_bound = forecast_df['mean_ci_upper'].values
            lower_bound = forecast_df['mean_ci_lower'].values
            
            return {
                'forecast': forecast_values.tolist(),
                'upper_bound': upper_bound.tolist(),
                'lower_bound': lower_bound.tolist(),
                'mean_forecast': float(forecast_values.mean()),
                'forecast_7d': float(forecast_values[6]) if len(forecast_values) > 6 else float(forecast_values[-1]),
                'forecast_30d': float(forecast_values[-1]),
                'model_order': best_order,
                'aic': float(best_aic),
                'confidence_width': float((upper_bound - lower_bound).mean())
            }
            
        except Exception as e:
            print(f"ARIMA预测错误: {e}")
            # Fallback：简单移动平均
            mean_price = np.mean(self.prices[-30:])
            std_price = np.std(self.prices[-30:])
            forecast = [mean_price] * steps
            
            return {
                'forecast': forecast,
                'upper_bound': [mean_price + 1.96 * std_price] * steps,
                'lower_bound': [mean_price - 1.96 * std_price] * steps,
                'mean_forecast': mean_price,
                'forecast_7d': mean_price,
                'forecast_30d': mean_price,
                'model_order': (0, 0, 0),
                'aic': 0,
                'confidence_width': 1.96 * std_price * 2
            }
    
    def garch_volatility(self, forecast_days=30):
        """改进的GARCH波动率预测（含真实IV和IV Skew）"""
        try:
            # 计算收益率
            returns = pd.Series(self.prices).pct_change().dropna() * 100
            
            # GARCH(1,1)模型
            model = arch_model(returns, vol='Garch', p=1, q=1)
            fitted = model.fit(disp='off')
            
            # 预测波动率
            forecast = fitted.forecast(horizon=forecast_days)
            predicted_vol = np.sqrt(forecast.variance.values[-1, :])
            predicted_vol_annual = predicted_vol / 100 * np.sqrt(252)
            
            # 历史波动率（年化）
            current_vol_annual = returns.std() / 100 * np.sqrt(252)
            
            # 尝试从真实期权链获取IV
            iv_skew = None
            implied_vol_atm = None
            
            try:
                stock = yf.Ticker(self.ticker)
                expiration_dates = stock.options
                
                if len(expiration_dates) > 0:
                    # 获取最近30天左右的到期日
                    target_dte = 30
                    selected_exp = None
                    min_diff = float('inf')
                    
                    for exp_str in expiration_dates:
                        exp_date = pd.to_datetime(exp_str)
                        dte = (exp_date - pd.Timestamp.now()).days
                        
                        if abs(dte - target_dte) < min_diff and dte > 0:
                            min_diff = abs(dte - target_dte)
                            selected_exp = exp_str
                    
                    if selected_exp:
                        chain = stock.option_chain(selected_exp)
                        calls = chain.calls
                        puts = chain.puts
                        
                        current_price = self.prices[-1]
                        
                        # 获取ATM期权的IV
                        calls['moneyness_diff'] = abs(calls['strike'] - current_price)
                        atm_call = calls.loc[calls['moneyness_diff'].idxmin()]
                        
                        if atm_call['impliedVolatility'] > 0:
                            implied_vol_atm = float(atm_call['impliedVolatility'])
                        
                        # 构建IV Skew
                        # OTM Call (105% strike)
                        otm_call_strikes = calls[calls['strike'] > current_price * 1.04]
                        if not otm_call_strikes.empty:
                            otm_call = otm_call_strikes.iloc[0]
                            iv_otm_call = float(otm_call['impliedVolatility']) if otm_call['impliedVolatility'] > 0 else implied_vol_atm * 0.95
                        else:
                            iv_otm_call = implied_vol_atm * 0.95 if implied_vol_atm else current_vol_annual * 1.10
                        
                        # OTM Put (95% strike)
                        otm_put_strikes = puts[puts['strike'] < current_price * 0.96]
                        if not otm_put_strikes.empty:
                            otm_put = otm_put_strikes.iloc[-1]
                            iv_otm_put = float(otm_put['impliedVolatility']) if otm_put['impliedVolatility'] > 0 else implied_vol_atm * 1.10
                        else:
                            iv_otm_put = implied_vol_atm * 1.10 if implied_vol_atm else current_vol_annual * 1.20
                        
                        if implied_vol_atm and implied_vol_atm > 0:
                            iv_skew = {
                                'atm': implied_vol_atm,
                                'otm_call': iv_otm_call,
                                'otm_put': iv_otm_put,
                                'skew_call': (iv_otm_call - implied_vol_atm) / implied_vol_atm * 100,
                                'skew_put': (iv_otm_put - implied_vol_atm) / implied_vol_atm * 100
                            }
            
            except Exception as e:
                print(f"获取真实IV失败: {e}")
            
            # 如果没有获取到真实IV，使用估计值
            if implied_vol_atm is None or implied_vol_atm <= 0:
                implied_vol_atm = current_vol_annual * 1.15  # 假设IV溢价15%
                iv_skew = {
                    'atm': implied_vol_atm,
                    'otm_call': implied_vol_atm * 0.95,
                    'otm_put': implied_vol_atm * 1.10,
                    'skew_call': -5.0,
                    'skew_put': 10.0
                }
            
            # 波动率错误定价（市场IV vs GARCH预测）
            vol_mispricing = (implied_vol_atm - np.mean(predicted_vol_annual)) / implied_vol_atm * 100
            
            # IV百分位（当前IV在历史分布中的位置）
            historical_vol_30d = returns[-30:].std() / 100 * np.sqrt(252)
            historical_vol_60d = returns[-60:].std() / 100 * np.sqrt(252)
            historical_vol_90d = returns[-90:].std() / 100 * np.sqrt(252)
            
            historical_vols = [historical_vol_30d, historical_vol_60d, historical_vol_90d]
            iv_percentile = sum(implied_vol_atm > hv for hv in historical_vols) / len(historical_vols) * 100
            
            return {
                'predicted_vol': float(np.mean(predicted_vol_annual)),
                'current_iv': float(implied_vol_atm),
                'historical_vol': float(current_vol_annual),
                'iv_skew': iv_skew,
                'vol_mispricing': float(vol_mispricing),
                'iv_percentile': float(iv_percentile),
                'forecast_vol': predicted_vol_annual.tolist(),
                'garch_params': {
                    'omega': float(fitted.params['omega']),
                    'alpha': float(fitted.params['alpha[1]']),
                    'beta': float(fitted.params['beta[1]'])
                }
            }
            
        except Exception as e:
            print(f"GARCH计算错误: {e}")
            # Fallback
            returns = pd.Series(self.prices).pct_change().dropna() * 100
            vol = returns.std() / 100 * np.sqrt(252)
            
            return {
                'predicted_vol': float(vol * 0.9),
                'current_iv': float(vol * 1.15),
                'historical_vol': float(vol),
                'iv_skew': {
                    'atm': vol * 1.15,
                    'otm_call': vol * 1.09,
                    'otm_put': vol * 1.27,
                    'skew_call': -5.0,
                    'skew_put': 10.0
                },
                'vol_mispricing': 15.0,
                'iv_percentile': 50.0,
                'forecast_vol': [vol] * forecast_days,
                'garch_params': {'omega': 0, 'alpha': 0, 'beta': 0}
            }
    
    def get_risk_free_rate(self):
        """获取无风险利率"""
        try:
            treasury = yf.Ticker("^IRX")  # 13周美国国债
            rate_data = treasury.history(period='5d')
            
            if not rate_data.empty:
                return rate_data['Close'].iloc[-1] / 100
        except:
            pass
        
        return 0.045  # 默认4.5%
    
    def black_scholes(self, S, K, T, r, sigma, option_type='call'):
        """Black-Scholes期权定价公式
        
        Args:
            S: 现价
            K: 行权价
            T: 到期时间（年）
            r: 无风险利率
            sigma: 波动率（年化）
            option_type: 'call' 或 'put'
        
        Returns:
            期权价格
        """
        if T <= 0:
            # 到期时的内在价值
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        # 避免除零和负波动率
        if sigma <= 0:
            sigma = 0.01
        
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            return max(price, 0.01)  # 最小价格0.01
            
        except Exception as e:
            print(f"BS定价错误: {e}")
            # Fallback to intrinsic value
            if option_type == 'call':
                return max(S - K, 0.01)
            else:
                return max(K - S, 0.01)
    
    def calculate_greeks(self, S, strikes, T, r, sigmas):
        """计算蝴蝶组合的Greeks
        
        Args:
            S: 现价
            strikes: [下翼, 中心, 上翼] 行权价
            T: 到期时间（年）
            r: 无风险利率
            sigmas: [下翼σ, 中心σ, 上翼σ]
        
        Returns:
            Greeks字典
        """
        def calculate_single_greeks(S, K, T, r, sigma):
            """计算单个Call期权的Greeks"""
            if T <= 0 or sigma <= 0:
                return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0}
            
            try:
                d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                
                # Delta
                delta = norm.cdf(d1)
                
                # Gamma
                gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
                
                # Vega (per 1% change in volatility)
                vega = S * norm.pdf(d1) * np.sqrt(T) / 100
                
                # Theta (per day)
                theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                         r * K * np.exp(-r*T) * norm.cdf(d2)) / 365
                
                return {
                    'delta': float(delta),
                    'gamma': float(gamma),
                    'vega': float(vega),
                    'theta': float(theta)
                }
            except:
                return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0}
        
        # 计算每腿的Greeks
        lower_greeks = calculate_single_greeks(S, strikes[0], T, r, sigmas[0])
        center_greeks = calculate_single_greeks(S, strikes[1], T, r, sigmas[1])
        upper_greeks = calculate_single_greeks(S, strikes[2], T, r, sigmas[2])
        
        # 蝴蝶组合：+1下翼 -2中间 +1上翼
        butterfly_greeks = {
            'delta': lower_greeks['delta'] - 2*center_greeks['delta'] + upper_greeks['delta'],
            'gamma': lower_greeks['gamma'] - 2*center_greeks['gamma'] + upper_greeks['gamma'],
            'vega': lower_greeks['vega'] - 2*center_greeks['vega'] + upper_greeks['vega'],
            'theta': lower_greeks['theta'] - 2*center_greeks['theta'] + upper_greeks['theta']
        }
        
        return {k: round(float(v), 4) for k, v in butterfly_greeks.items()}
    
    def design_butterfly(self, forecast_price, price_stability, volatility, iv_skew, dominant_period):
        """改进的蝴蝶期权设计（使用真实BS定价和IV Skew）
        
        Args:
            forecast_price: ARIMA预测价格
            price_stability: 价格稳定性指标
            volatility: GARCH预测的波动率
            iv_skew: IV偏斜数据
            dominant_period: 主导周期（天数）
        
        Returns:
            蝴蝶策略详情
        """
        current_price = self.prices[-1]
        
        # 确定行权价间隔
        if current_price < 50:
            strike_step = 2.5
        elif current_price < 100:
            strike_step = 5
        elif current_price < 200:
            strike_step = 5
        else:
            strike_step = 10
        
        # 中心行权价（基于ARIMA预测，四舍五入到strike_step）
        center_strike = round(forecast_price / strike_step) * strike_step
        
        # 翼宽（基于价格稳定性和主导周期）
        if price_stability < 8:
            wing_width = strike_step
        elif price_stability < 12:
            wing_width = strike_step * 2
        else:
            wing_width = strike_step * 3
        
        # 根据主导周期调整翼宽
        if dominant_period < 15:
            wing_width = max(strike_step, wing_width * 0.8)
        elif dominant_period > 45:
            wing_width = wing_width * 1.2
        
        wing_width = round(wing_width / strike_step) * strike_step
        
        lower_strike = center_strike - wing_width
        upper_strike = center_strike + wing_width
        
        # 到期时间（基于主导周期）
        if dominant_period < 20:
            dte = 21
        elif dominant_period < 40:
            dte = 30
        else:
            dte = 45
        
        T = dte / 365
        
        # 无风险利率
        r = self.get_risk_free_rate()
        
        # 根据行权价的钱性使用不同的波动率（IV Skew调整）
        def get_sigma_for_strike(strike, current_price, iv_skew_data):
            moneyness = strike / current_price
            
            if moneyness < 0.96:  # OTM Put区域
                return iv_skew_data.get('otm_put', volatility * 1.10)
            elif moneyness > 1.04:  # OTM Call区域
                return iv_skew_data.get('otm_call', volatility * 0.95)
            else:  # ATM区域
                return iv_skew_data.get('atm', volatility)
        
        sigma_lower = get_sigma_for_strike(lower_strike, current_price, iv_skew)
        sigma_center = get_sigma_for_strike(center_strike, current_price, iv_skew)
        sigma_upper = get_sigma_for_strike(upper_strike, current_price, iv_skew)
        
        # 计算各腿的理论价格（BS模型）
        lower_call_price = self.black_scholes(
            current_price, lower_strike, T, r, sigma_lower, 'call'
        )
        center_call_price = self.black_scholes(
            current_price, center_strike, T, r, sigma_center, 'call'
        )
        upper_call_price = self.black_scholes(
            current_price, upper_strike, T, r, sigma_upper, 'call'
        )
        
        # 理论净成本
        net_debit_theoretical = lower_call_price - 2 * center_call_price + upper_call_price
        
        # 加入Bid-Ask Spread（基于钱性估计）
        def estimate_spread(strike, current_price):
            moneyness = abs(strike / current_price - 1)
            
            if moneyness < 0.03:  # ATM
                return 0.05  # 5%
            elif moneyness < 0.08:  # Near ATM
                return 0.07  # 7%
            else:  # OTM
                return 0.10  # 10%
        
        spread_pct_lower = estimate_spread(lower_strike, current_price)
        spread_pct_center = estimate_spread(center_strike, current_price)
        spread_pct_upper = estimate_spread(upper_strike, current_price)
        
        # 实际执行成本（买入用Ask，卖出用Bid）
        lower_cost_actual = lower_call_price * (1 + spread_pct_lower / 2)
        center_credit_actual = center_call_price * (1 - spread_pct_center / 2)
        upper_cost_actual = upper_call_price * (1 + spread_pct_upper / 2)
        
        net_debit_actual = (lower_cost_actual - 2 * center_credit_actual + upper_cost_actual)
        
        # 确保净成本为正
        net_debit_actual = max(0.10, net_debit_actual)
        
        # 最大收益和最大损失
        max_profit = wing_width - net_debit_actual
        max_loss = net_debit_actual
        
        # 盈亏平衡点
        breakeven_lower = lower_strike + net_debit_actual
        breakeven_upper = upper_strike - net_debit_actual
        
        # 计算Greeks
        greeks = self.calculate_greeks(
            current_price,
            [lower_strike, center_strike, upper_strike],
            T, r,
            [sigma_lower, sigma_center, sigma_upper]
        )
        
        # 预期收益概率（基于ARIMA预测）
        # 假设价格在预测区间内均匀分布（简化）
        prob_profit = 0
        if breakeven_lower < forecast_price < breakeven_upper:
            prob_profit = 0.68  # 68% (1 sigma)
        elif lower_strike < forecast_price < upper_strike:
            prob_profit = 0.50
        else:
            prob_profit = 0.30
        
        return {
            'center_strike': float(center_strike),
            'lower_strike': float(lower_strike),
            'upper_strike': float(upper_strike),
            'wing_width': float(wing_width),
            'dte': int(dte),
            'lower_cost': round(float(lower_cost_actual), 2),
            'center_credit': round(float(center_credit_actual), 2),
            'upper_cost': round(float(upper_cost_actual), 2),
            'net_debit': round(float(net_debit_actual), 2),
            'max_profit': round(float(max(0.01, max_profit)), 2),
            'max_loss': round(float(max_loss), 2),
            'profit_ratio': round(float(max_profit / max_loss), 2) if max_loss > 0 else 0,
            'breakeven_lower': round(float(breakeven_lower), 2),
            'breakeven_upper': round(float(breakeven_upper), 2),
            'prob_profit': round(float(prob_profit * 100), 1),
            'risk_free_rate': round(float(r * 100), 2),
            'greeks': greeks,
            'spreads': {
                'lower': round(spread_pct_lower * 100, 1),
                'center': round(spread_pct_center * 100, 1),
                'upper': round(spread_pct_upper * 100, 1)
            }
        }

    def calculate_strategy_score(self, fourier, arima, garch, butterfly, price_stability):
        """计算蝴蝶策略的综合评分（0-100）"""
    
        # 因子1：价格预测匹配度（35%权重）
        forecast_center_diff = abs(arima['mean_forecast'] - butterfly['center_strike'])
        price_match_score = max(0, 100 - (forecast_center_diff / arima['mean_forecast'] * 500))
    
        # 因子2：波动率错误定价（30%权重）
        # IV被高估（正值）对卖方策略有利
        vol_score = min(100, max(0, garch['vol_mispricing'] * 5 + 50))
    
        # 因子3：价格稳定性（20%权重）
        # 稳定性越高，蝴蝶策略越有利
        stability_score = max(0, 100 - price_stability * 5)
    
        # 因子4：傅立叶周期对齐（15%权重）
        trend_dir = fourier['trend_direction']
        bf_type = fourier['butterfly_type']
        cycle_pos = fourier['cycle_position']
    
        if (bf_type == 'CALL' and trend_dir == 'UP' and cycle_pos == 'TROUGH') or \
           (bf_type == 'PUT' and trend_dir == 'DOWN' and cycle_pos == 'PEAK') or \
           (bf_type == 'IRON' and trend_dir == 'FLAT'):
            fourier_score = 100
        elif trend_dir == 'FLAT':
            fourier_score = 80
        else:
            fourier_score = 50
    
        # 考虑周期强度
        if fourier.get('period_strength', 0) > 10:
            fourier_score *= 1.1  # 周期明显，加分
    
        fourier_score = min(100, fourier_score)
    
        # 加权综合
        total_score = (
            price_match_score * 0.35 +
            vol_score * 0.30 +
            stability_score * 0.20 +
            fourier_score * 0.15
        )
    
        # Greeks惩罚：Delta不够中性会降低评分
        delta_penalty = min(10, abs(butterfly['greeks']['delta']) * 50)
        total_score -= delta_penalty
    
        total_score = max(0, min(100, total_score))
    
        return {
            'total': round(total_score, 1),
            'components': {
                'price_match': round(price_match_score, 1),
                'vol_mispricing': round(vol_score, 1),
                'stability': round(stability_score, 1),
                'fourier_align': round(fourier_score, 1),
                'delta_penalty': round(delta_penalty, 1)
            },
            'recommendation': self._get_recommendation(total_score, butterfly['profit_ratio']),
            'confidence_level': self._get_confidence_level(total_score)
        }
    def _get_recommendation(self, score, profit_ratio):
        """根据评分和盈亏比给出建议"""
        if score >= 75 and profit_ratio > 2:
            return 'STRONG_BUY'
        elif score >= 60 and profit_ratio > 1.5:
            return 'BUY'
        elif score >= 45:
            return 'NEUTRAL'
        else:
            return 'AVOID'

    def _get_confidence_level(self, score):
        """评估置信度水平"""
        if score >= 80:
            return 'HIGH'
        elif score >= 60:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _assess_risk_level(self, price_stability, vol_mispricing, greeks):
        """风险评估（考虑Greeks）"""
        # 基础风险
        if price_stability < 8 and vol_mispricing > 15:
            base_risk = 'LOW'
        elif price_stability < 15 and vol_mispricing > 5:
            base_risk = 'MEDIUM'
        else:
            base_risk = 'HIGH'
        
        # Greeks风险调整
        if abs(greeks['delta']) > 0.15:  # Delta不够中性
            if base_risk == 'LOW':
                base_risk = 'MEDIUM'
            elif base_risk == 'MEDIUM':
                base_risk = 'HIGH'
        
        # Vega风险（负Vega意味着做空波动率）
        if greeks['vega'] > -0.5:  # Vega应该是负 of
            if base_risk == 'LOW':
                base_risk = 'MEDIUM'
        
        return base_risk

    def full_analysis(self):
        """完整分析（改进版）"""
        self.fetch_data()
        
        # 傅立叶分析
        fourier_result = self.fourier_analysis()
        
        # ARIMA预测
        arima_result = self.arima_forecast()
        
        # GARCH波动率
        garch_result = self.garch_volatility()
        
        # 计算价格稳定性（预测区间宽度）
        price_range = (max(arima_result['upper_bound']) - 
                       min(arima_result['lower_bound']))
        price_stability = price_range / arima_result['mean_forecast'] * 100
        
        # 设计蝴蝶期权
        butterfly = self.design_butterfly(
            arima_result['mean_forecast'],
            price_stability,
            garch_result['predicted_vol'],
            garch_result['iv_skew'],
            fourier_result['dominant_period_days']
        )
        
        # 综合评分
        score = self.calculate_strategy_score(
            fourier_result,
            arima_result,
            garch_result,
            butterfly,
            price_stability
        )
        
        # 交易信号
        signals = {
            'price_stability': price_stability < 12,
            'vol_mispricing': garch_result['vol_mispricing'] > 10,
            'trend_clear': fourier_result['trend_direction'] != 'FLAT',
            'cycle_aligned': (
                (fourier_result['trend_direction'] == 'UP' and 
                 fourier_result['cycle_position'] == 'TROUGH') or
                (fourier_result['trend_direction'] == 'DOWN' and 
                 fourier_result['cycle_position'] == 'PEAK')
            ),
            'delta_neutral': abs(butterfly['greeks']['delta']) < 0.10,
            'iv_high': garch_result['iv_percentile'] > 60
        }
        
        # 风险评估
        risk_level = self._assess_risk_level(
            price_stability,
            garch_result['vol_mispricing'],
            butterfly['greeks']
        )
        
        # 准备图表数据
        timestamps = self.data.index.tolist()
        chart_data = self.prepare_chart_data(
            timestamps,
            fourier_result,
            arima_result,
            garch_result
        )
        
        # 交易建议
        trade_suggestion = self._generate_trade_suggestion(
            score,
            butterfly,
            signals,
            risk_level
        )
        
        return {
            'ticker': self.ticker,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': round(float(self.prices[-1]), 2),
            'forecast_price': round(arima_result['mean_forecast'], 2),
            'upper_bound': round(max(arima_result['upper_bound']), 2),
            'lower_bound': round(min(arima_result['lower_bound']), 2),
            'price_stability': round(price_stability, 1),
            'fourier': fourier_result,
            'arima': arima_result,
            'garch': garch_result,
            'butterfly': butterfly,
            'signals': signals,
            'risk_level': risk_level,
            'score': score,
            'trade_suggestion': trade_suggestion,
            'chart_data': chart_data
        }

    def _generate_trade_suggestion(self, score, butterfly, signals, risk_level):
        """生成具体交易建议"""
        recommendation = score['recommendation']
        
        suggestion = {
            'action': recommendation,
            'position_size': 'SMALL' if risk_level == 'HIGH' else 'MEDIUM' if risk_level == 'MEDIUM' else 'STANDARD',
            'entry_timing': 'IMMEDIATE' if score['total'] > 70 else 'WAIT_FOR_PULLBACK',
            'stop_loss': round(butterfly['net_debit'] * 1.5, 2),
            'take_profit': round(butterfly['max_profit'] * 0.7, 2),
            'hold_until': f"{butterfly['dte']} days or 70% max profit",
            'key_risks': []
        }
        
        # 关键风险提示
        if not signals['delta_neutral']:
            suggestion['key_risks'].append('Delta不够中性，有方向性风险')
        
        if not signals['price_stability']:
            suggestion['key_risks'].append('价格波动较大，蝴蝶策略风险增加')
        
        if not signals['iv_high']:
            suggestion['key_risks'].append('IV不在高位，卖期权优势不明显')
        
        if butterfly['profit_ratio'] < 1.5:
            suggestion['key_risks'].append('盈亏比偏低，风险收益不对称')
        
        return suggestion

    def prepare_chart_data(self, timestamps, fourier, arima, garch):
        """准备前端图表数据"""
        # 傅立叶分解数据
        fourier_data = []
        start_idx = max(0, len(self.prices) - 120)
        
        for i in range(start_idx, len(self.prices)):
            if i < len(fourier['low_freq_signal']) and i < len(fourier['mid_freq_signal']):
                fourier_data.append({
                    'date': timestamps[i].strftime('%m/%d'),
                    'actual': round(float(self.prices[i]), 2),
                    'lowFreq': round(float(fourier['low_freq_signal'][i]), 2),
                    'midFreq': round(float(fourier['mid_freq_signal'][i]), 2)
                })
        
        # 价格预测数据
        price_forecast_data = []
        recent_start = max(0, len(self.prices) - 60)
        
        # 历史数据
        for i in range(recent_start, len(self.prices)):
            price_forecast_data.append({
                'date': timestamps[i].strftime('%m/%d'),
                'actual': round(float(self.prices[i]), 2),
                'forecast': None,
                'upper': None,
                'lower': None
            })
        
        # 预测数据
        forecast_len = min(30, len(arima['forecast']))
        for i in range(forecast_len):
            future_date = (timestamps[-1] + timedelta(days=i+1)).strftime('%m/%d')
            price_forecast_data.append({
                'date': future_date,
                'actual': None,
                'forecast': round(float(arima['forecast'][i]), 2),
                'upper': round(float(arima['upper_bound'][i]), 2),
                'lower': round(float(arima['lower_bound'][i]), 2)
            })
        
        # 波动率数据
        vol_data = []
        
        # 历史实现波动率（最近30天）
        returns = pd.Series(self.prices).pct_change().dropna() * 100
        rolling_vol = returns.rolling(window=30).std() / 100 * np.sqrt(252)
        
        recent_vol_start = max(0, len(rolling_vol) - 30)
        
        for i in range(recent_vol_start, len(rolling_vol)):
            if not np.isnan(rolling_vol.iloc[i]):
                date = timestamps[i + 1].strftime('%m/%d')
                vol_data.append({
                    'date': date,
                    'realized': round(float(rolling_vol.iloc[i]), 4),
                    'predicted': None
                })
        
        # 预测波动率
        forecast_vol_len = min(30, len(garch['forecast_vol']))
        for i in range(forecast_vol_len):
            future_date = (timestamps[-1] + timedelta(days=i+1)).strftime('%m/%d')
            vol_data.append({
                'date': future_date,
                'realized': None,
                'predicted': round(float(garch['forecast_vol'][i]), 4)
            })
        
        # 功率谱数据
        spectrum_data = []
        for period_info in fourier['dominant_periods'][:5]:
            period_days = period_info['period']
            
            if period_days < 365:
                period_label = f"{int(period_days)}天"
            else:
                period_label = f"{period_days/365:.1f}年"
            
            spectrum_data.append({
                'period': period_label,
                'power': round(float(period_info['power']), 2),
                'powerPct': round(float(period_info.get('power_pct', 0)), 2),
                'periodDays': round(float(period_days), 1)
            })
        
        return {
            'fourier': fourier_data,
            'price_forecast': price_forecast_data,
            'volatility': vol_data,
            'spectrum': spectrum_data
        }

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """分析接口"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', 'AAPL').upper()
        # 验证ticker格式
        if not ticker or len(ticker) > 10:
            return jsonify({
                'success': False,
                'error': '无效的股票代码'
            }), 400
    
        analyzer = ButterflyAnalyzer(ticker)
        result = analyzer.full_analysis()
    
        return jsonify({
            'success': True,
            'data': result
        })

    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    
    except Exception as e:
        import traceback
        print(f"分析错误: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'分析过程发生错误: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'version': '2.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/tickers', methods=['GET'])
def get_popular_tickers():
    """获取常用股票列表"""
    popular_tickers = [
    {'symbol': 'AAPL', 'name': 'Apple Inc.'},
    {'symbol': 'MSFT', 'name': 'Microsoft Corporation'},
    {'symbol': 'GOOGL', 'name': 'Alphabet Inc.'},
    {'symbol': 'AMZN', 'name': 'Amazon.com Inc.'},
    {'symbol': 'TSLA', 'name': 'Tesla Inc.'},
    {'symbol': 'NVDA', 'name': 'NVIDIA Corporation'},
    {'symbol': 'META', 'name': 'Meta Platforms Inc.'},
    {'symbol': 'SPY', 'name': 'S&P 500 ETF'},
    {'symbol': 'QQQ', 'name': 'Nasdaq-100 ETF'},
    ]
    return jsonify({
    'success': True,
    'tickers': popular_tickers
    })

if __name__ == '__main__':
    import sys
    import io
    
    # 设置标准输出为 UTF-8 编码
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("="*60)
    print("🚀 ARIMA-GARCH蝴蝶期权分析后端启动 (改进版 v2.0)")
    print("="*60)
    print("📊 健康检查: http://localhost:5000/api/health")
    print("💡 分析接口: POST http://localhost:5000/api/analyze")
    print("   请求示例: {'ticker': 'AAPL'}")
    print("📈 常用股票: GET http://localhost:5000/api/tickers")
    print("="*60)
    print("\n主要改进:")
    print("✅ 真正的去趋势傅立叶分析（VWAP基准）")
    print("✅ ARIMA自动参数选择")
    print("✅ 真实期权链IV + IV Skew")
    print("✅ Black-Scholes精确定价")
    print("✅ 完整的Greeks计算")
    print("✅ 多因子综合评分系统")
    print("✅ 智能交易建议生成")
    print("="*60)
    print("\n正在启动服务器...")
    app.run(debug=True, port=5000, host='0.0.0.0')