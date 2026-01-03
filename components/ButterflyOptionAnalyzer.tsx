import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Area, AreaChart, BarChart, Bar } from 'recharts';
import { TrendingUp, Activity, DollarSign, AlertTriangle, CheckCircle, Play, Waves, TrendingDown, Award, Shield, Target, Info } from 'lucide-react';
import { Helmet } from 'react-helmet-async';

// Use a typed constant for the custom element to bypass JSX.IntrinsicElements check
const TvMiniChart = 'tv-mini-chart' as any;

const ButterflyOptionAnalyzer: React.FC = () => {
  const [ticker, setTicker] = useState<string>('');
  const [analyzedTicker, setAnalyzedTicker] = useState<string>('');
  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  // 计算动态标题
  const pageTitle = (() => {
    if (results && ticker.toUpperCase() === analyzedTicker) {
      return `${analyzedTicker} 蝶式期权分析 | ButterQuantDL v2.1`;
    } else if (ticker) {
      return `${ticker.toUpperCase()} - ButterQuantDL分析器 v2.1`;
    } else {
      return `ButterQuantDL分析器 v2.1`;
    }
  })();

  const runAnalysis = async () => {
    if (!ticker) return;
    setAnalyzing(true);
    setError(null);
    setResults(null);

    try {
      const response = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ticker: ticker })
      });

      const data = await response.json();

      if (data.success) {
        setResults(data.data);
        setAnalyzedTicker(ticker.toUpperCase());
      } else {
        setError(data.error || '分析失败');
      }
    } catch (err: any) {
      setError(`连接后端失败: ${err.message}. 请确保Python后端正在运行 (http://localhost:5000)`);
    } finally {
      setAnalyzing(false);
    }
  };

  const getTrendIcon = (direction: string) => {
    if (direction === 'UP') return <TrendingUp className="w-6 h-6 text-green-500" />;
    if (direction === 'DOWN') return <TrendingDown className="w-6 h-6 text-red-500" />;
    return <Activity className="w-6 h-6 text-gray-500" />;
  };

  const getButterflyColor = (type: string) => {
    if (type === 'CALL') return 'text-green-600 bg-green-50 border-green-200';
    if (type === 'PUT') return 'text-red-600 bg-red-50 border-red-200';
    return 'text-blue-600 bg-blue-50 border-blue-200';
  };

  const getSignalIcon = (value: boolean) => {
    return value ? <CheckCircle className="w-5 h-5 text-green-500" /> : <AlertTriangle className="w-5 h-5 text-red-500" />;
  };

  const getRecommendationStyle = (recommendation: string) => {
    const styles: Record<string, { color: string; bg: string; border: string; text: string }> = {
      'STRONG_BUY': { color: 'text-green-700', bg: 'bg-green-100', border: 'border-green-300', text: '强烈买入' },
      'BUY': { color: 'text-green-600', bg: 'bg-green-50', border: 'border-green-200', text: '买入' },
      'NEUTRAL': { color: 'text-yellow-600', bg: 'bg-yellow-50', border: 'border-yellow-200', text: '中性观望' },
      'AVOID': { color: 'text-red-600', bg: 'bg-red-50', border: 'border-red-200', text: '避免' }
    };
    return styles[recommendation] || styles['NEUTRAL'];
  };

  const getLegTypes = (type: string) => {
    const isIron = type === 'IRON';
    // Iron Butterfly: Buy Lower Put, Sell Straddle (Call+Put), Buy Upper Call
    const lower = isIron ? 'Put' : (type === 'CALL' ? 'Call' : 'Put');
    const upper = isIron ? 'Call' : (type === 'CALL' ? 'Call' : 'Put');
    const center = isIron ? 'Straddle (Call + Put)' : (type === 'CALL' ? '2 Calls' : '2 Puts');
    
    return { lower, center, upper, isIron };
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-50">
      <Helmet>
        <title>{pageTitle}</title>
      </Helmet>

      {/* 头部 */}
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <div className="flex flex-col md:flex-row md:items-start justify-between gap-6 mb-6">
          <div className="flex-1">
            <h1 className="text-3xl font-bold text-gray-800 flex items-center gap-2">
              <Waves className="w-8 h-8 text-blue-600" />
              ButterQuantDL分析器
              <span className="text-sm font-normal text-blue-600 bg-blue-100 px-2 py-1 rounded ml-2">v2.1</span>
            </h1>
            <p className="text-gray-600 mt-2">通过频域分析判断趋势方向，自动选择Call/Put/Iron Butterfly策略</p>
            <div className="flex flex-wrap gap-2 mt-2">
              <span className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded">✨ 真实IV Skew</span>
              <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded">✨ BS精确定价</span>
              <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded">✨ Greeks计算</span>
              <span className="text-xs bg-yellow-100 text-yellow-700 px-2 py-1 rounded">✨ 智能评分</span>
            </div>
          </div>
          
          <div className="w-full md:w-[350px] shrink-0 h-[150px] bg-gray-50 rounded overflow-hidden">
            <TvMiniChart
              symbol={ticker || "AAPL"}
              line-chart-type="Baseline"
              theme="light"
              autosize="false"
              width="100%"
              height="100%"
            ></TvMiniChart>
          </div>
        </div>

        <div className="flex gap-4 items-end">
          <div className="flex-1">
            <label className="block text-sm font-medium text-gray-700 mb-2">股票代码</label>
            <input
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="输入美股代码 (如: AAPL, TSLA, SPY)"
              onKeyPress={(e) => e.key === 'Enter' && runAnalysis()}
            />
          </div>
          <button
            onClick={runAnalysis}
            disabled={analyzing || !ticker}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 flex items-center gap-2 transition-all"
          >
            {analyzing ? (
              <>
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                分析中...
              </>
            ) : (
              <>
                <Play className="w-5 h-5" />
                开始分析
              </>
            )}
          </button>
        </div>

        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            <p className="font-semibold">❌ 错误</p>
            <p className="text-sm mt-1">{error}</p>
          </div>
        )}
      </div>

      {results && (
        <>
          {/* 综合评分卡片 */}
          {results.score && (
            <div className="bg-gradient-to-r from-purple-500 to-indigo-600 rounded-lg shadow-lg p-6 mb-6 text-white">
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <Award className="w-8 h-8" />
                    <h2 className="text-2xl font-bold">策略综合评分</h2>
                  </div>
                  <p className="text-purple-100 text-sm">多因子量化评估系统</p>
                </div>
                <div className="text-right">
                  <div className="text-6xl font-bold">{results.score.total}</div>
                  <div className="text-xl text-purple-200">/ 100</div>
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
                <div className="bg-white/10 backdrop-blur rounded-lg p-3">
                  <div className="text-xs text-purple-200 mb-1">价格匹配度</div>
                  <div className="text-2xl font-bold">{results.score.components.price_match}</div>
                  <div className="text-xs text-purple-200 mt-1">35%权重</div>
                </div>
                <div className="bg-white/10 backdrop-blur rounded-lg p-3">
                  <div className="text-xs text-purple-200 mb-1">波动率错配</div>
                  <div className="text-2xl font-bold">{results.score.components.vol_mispricing}</div>
                  <div className="text-xs text-purple-200 mt-1">30%权重</div>
                </div>
                <div className="bg-white/10 backdrop-blur rounded-lg p-3">
                  <div className="text-xs text-purple-200 mb-1">价格稳定性</div>
                  <div className="text-2xl font-bold">{results.score.components.stability}</div>
                  <div className="text-xs text-purple-200 mt-1">20%权重</div>
                </div>
                <div className="bg-white/10 backdrop-blur rounded-lg p-3">
                  <div className="text-xs text-purple-200 mb-1">周期对齐</div>
                  <div className="text-2xl font-bold">{results.score.components.fourier_align}</div>
                  <div className="text-xs text-purple-200 mt-1">15%权重</div>
                </div>
              </div>

              {results.score.components.delta_penalty > 0 && (
                <div className="mt-4 bg-yellow-500/20 border border-yellow-300/30 rounded p-2 text-sm">
                  ⚠️ Delta中性惩罚: -{results.score.components.delta_penalty} 分
                </div>
              )}

              <div className="mt-4 flex items-center justify-between">
                <div className={`px-4 py-2 rounded-lg font-semibold ${getRecommendationStyle(results.score.recommendation).bg} ${getRecommendationStyle(results.score.recommendation).color} border ${getRecommendationStyle(results.score.recommendation).border}`}>
                  📊 推荐: {getRecommendationStyle(results.score.recommendation).text}
                </div>
                <div className="text-sm">
                  置信度: <span className="font-bold">{results.score.confidence_level}</span>
                </div>
              </div>
            </div>
          )}

          {/* 核心指标卡片 */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-white rounded-lg shadow p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600">价格稳定性</span>
                {getSignalIcon(results.signals.price_stability)}
              </div>
              <p className="text-2xl font-bold text-gray-800">{results.price_stability}%</p>
              <p className="text-xs text-gray-500 mt-1">波动幅度 (越小越好)</p>
            </div>

            <div className="bg-white rounded-lg shadow p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600">波动率错配</span>
                {getSignalIcon(results.signals.vol_mispricing)}
              </div>
              <p className="text-2xl font-bold text-gray-800">
                {results.garch.vol_mispricing > 0 ? '+' : ''}{results.garch.vol_mispricing.toFixed(1)}%
              </p>
              <p className="text-xs text-gray-500 mt-1">IV高于预测 (做多蝴蝶)</p>
            </div>

            <div className="bg-white rounded-lg shadow p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600">盈亏比</span>
                <DollarSign className="w-5 h-5 text-green-500" />
              </div>
              <p className="text-2xl font-bold text-gray-800">{results.butterfly.profit_ratio.toFixed(1)}:1</p>
              <p className="text-xs text-gray-500 mt-1">最大收益/风险</p>
            </div>

            <div className="bg-white rounded-lg shadow p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600">风险等级</span>
                <AlertTriangle className="w-5 h-5 text-yellow-500" />
              </div>
              <p className={`text-2xl font-bold ${results.risk_level === 'LOW' ? 'text-green-600' : results.risk_level === 'MEDIUM' ? 'text-yellow-600' : 'text-red-600'}`}>
                {results.risk_level}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                {results.score ? `评分: ${results.score.total}分` : `置信度: ${results.confidence}%`}
              </p>
            </div>
          </div>

          {/* Greeks指标卡片 */}
          {results.butterfly.greeks && (
            <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
              <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                <Shield className="w-6 h-6 text-indigo-600" />
                Greeks 风险指标
              </h2>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                  <div className="text-xs text-gray-600 mb-1">Delta (方向性)</div>
                  <div className={`text-2xl font-bold ${Math.abs(results.butterfly.greeks.delta) < 0.10 ? 'text-green-600' : 'text-yellow-600'}`}>
                    {results.butterfly.greeks.delta.toFixed(4)}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {Math.abs(results.butterfly.greeks.delta) < 0.10 ? '✅ 中性' : '⚠️ 有方向性'}
                  </div>
                </div>

                <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
                  <div className="text-xs text-gray-600 mb-1">Gamma (凸性)</div>
                  <div className="text-2xl font-bold text-gray-800">
                    {results.butterfly.greeks.gamma.toFixed(4)}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {results.butterfly.greeks.gamma > 0 ? '在中心区域正Gamma' : '负Gamma'}
                  </div>
                </div>

                <div className="p-4 bg-pink-50 rounded-lg border border-pink-200">
                  <div className="text-xs text-gray-600 mb-1">Vega (波动率)</div>
                  <div className={`text-2xl font-bold ${results.butterfly.greeks.vega < 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {results.butterfly.greeks.vega.toFixed(4)}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {results.butterfly.greeks.vega < 0 ? '✅ 做空波动率' : '⚠️ 做多波动率'}
                  </div>
                </div>

                <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                  <div className="text-xs text-gray-600 mb-1">Theta (时间衰减)</div>
                  <div className={`text-2xl font-bold ${results.butterfly.greeks.theta > 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {results.butterfly.greeks.theta.toFixed(4)}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    每日 {results.butterfly.greeks.theta > 0 ? '+' : ''}{results.butterfly.greeks.theta.toFixed(2)}
                  </div>
                </div>
              </div>

              <div className="mt-4 p-4 bg-indigo-50 rounded-lg border border-indigo-200">
                <h3 className="font-semibold text-indigo-900 mb-2 flex items-center gap-2">
                  <Info className="w-4 h-4" />
                  Greeks 解读
                </h3>
                <ul className="text-sm text-indigo-800 space-y-1">
                  <li>• <strong>Delta ≈ 0</strong>: 蝴蝶策略应该方向中性，不受价格小幅波动影响</li>
                  <li>• <strong>Gamma &gt; 0</strong>: 在中心行权价附近有正Gamma，价格接近中心时获利加速</li>
                  <li>• <strong>Vega &lt; 0</strong>: 做空波动率，IV下降时获利（适合高IV时入场）</li>
                  <li>• <strong>Theta &gt; 0</strong>: 时间是朋友，每天赚取时间价值衰减</li>
                </ul>
              </div>
            </div>
          )}

          {/* IV Skew可视化 */}
          {results.garch.iv_skew && (
            <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
              <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                <Activity className="w-6 h-6 text-purple-600" />
                IV Skew 波动率偏斜
              </h2>

              <div className="grid grid-cols-3 gap-4 mb-4">
                <div className="p-4 bg-red-50 rounded-lg border border-red-200 text-center">
                  <div className="text-xs text-gray-600 mb-1">OTM Put (95%)</div>
                  <div className="text-2xl font-bold text-gray-800">
                    {(results.garch.iv_skew.otm_put * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-red-600 mt-1">
                    {results.garch.iv_skew.skew_put > 0 ? '+' : ''}{results.garch.iv_skew.skew_put.toFixed(1)}% vs ATM
                  </div>
                </div>

                <div className="p-4 bg-blue-50 rounded-lg border border-blue-200 text-center">
                  <div className="text-xs text-gray-600 mb-1">ATM (100%)</div>
                  <div className="text-2xl font-bold text-blue-600">
                    {(results.garch.iv_skew.atm * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-500 mt-1">基准IV</div>
                </div>

                <div className="p-4 bg-green-50 rounded-lg border border-green-200 text-center">
                  <div className="text-xs text-gray-600 mb-1">OTM Call (105%)</div>
                  <div className="text-2xl font-bold text-gray-800">
                    {(results.garch.iv_skew.otm_call * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-green-600 mt-1">
                    {results.garch.iv_skew.skew_call > 0 ? '+' : ''}{results.garch.iv_skew.skew_call.toFixed(1)}% vs ATM
                  </div>
                </div>
              </div>

              <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
                <h3 className="font-semibold text-purple-900 mb-2">📊 Skew含义</h3>
                <ul className="text-sm text-purple-800 space-y-1">
                  <li>• <strong>Put侧IV更高</strong>: 市场对下跌保护需求强（恐慌溢价）</li>
                  <li>• <strong>Call侧IV更低</strong>: 看涨期权相对便宜</li>
                  <li>• <strong>蝴蝶策略影响</strong>: 不同行权价使用不同IV，定价更精确</li>
                  <li>• <strong>IV百分位</strong>: {results.garch.iv_percentile.toFixed(0)}% 
                    {results.garch.iv_percentile > 75 ? ' (高位，适合卖期权)' : results.garch.iv_percentile < 25 ? ' (低位，不适合卖期权)' : ' (中等水平)'}
                  </li>
                </ul>
              </div>
            </div>
          )}

          {/* 傅立叶分析结果 */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className={`rounded-lg shadow p-4 border-2 ${getButterflyColor(results.fourier.butterfly_type)}`}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">策略类型</span>
                {getTrendIcon(results.fourier.trend_direction)}
              </div>
              <p className="text-3xl font-bold mb-2">{results.fourier.butterfly_type} Butterfly</p>
              <p className="text-xs opacity-75">{results.fourier.strategy_reason}</p>
            </div>

            <div className="bg-white rounded-lg shadow p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600">低频趋势</span>
                {getTrendIcon(results.fourier.trend_direction)}
              </div>
              <p className="text-2xl font-bold text-gray-800">
                {results.fourier.trend_direction === 'UP' ? '上涨' : results.fourier.trend_direction === 'DOWN' ? '下跌' : '平稳'}
              </p>
              <p className="text-xs text-gray-500 mt-1">斜率: {results.fourier.trend_slope.toFixed(4)}</p>
            </div>

            <div className="bg-white rounded-lg shadow p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600">中频周期位置</span>
                <Waves className="w-5 h-5 text-blue-500" />
              </div>
              <p className="text-2xl font-bold text-gray-800">
                {results.fourier.cycle_position === 'PEAK' ? '波峰' : '波谷'}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                主周期: {results.fourier.dominant_periods.slice(0, 2).map((p: any) => Math.round(p.period)).join(', ')}天
              </p>
            </div>
          </div>

          {/* 傅立叶分解图 */}
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <Waves className="w-6 h-6 text-purple-600" />
              傅立叶变换频域分解
            </h2>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={results.chart_data.fourier}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} interval={Math.floor(results.chart_data.fourier.length / 10)} />
                <YAxis domain={['auto', 'auto']} />
                <Tooltip />
                <Legend />

                <Line type="monotone" dataKey="actual" stroke="#3b82f6" strokeWidth={1.5} dot={false} name="原始价格" />
                <Line type="monotone" dataKey="lowFreq" stroke="#ef4444" strokeWidth={3} dot={false} name="低频趋势 (>60天)" />
                <Line type="monotone" dataKey="midFreq" stroke="#10b981" strokeWidth={2} dot={false} strokeDasharray="5 5" name="中频周期 (5-60天)" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* 功率谱图 */}
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <Activity className="w-6 h-6 text-green-600" />
              功率谱密度 - 主要周期识别
            </h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={results.chart_data.spectrum}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="period" tick={{ fontSize: 12 }} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="power" fill="#10b981" name="功率" />
                {results.chart_data.spectrum[0]?.powerPct && (
                  <Bar dataKey="powerPct" fill="#8b5cf6" name="功率占比%" />
                )}
              </BarChart>
            </ResponsiveContainer>

            <div className="mt-4 p-4 bg-green-50 rounded-lg border border-green-200">
              <h3 className="font-semibold text-green-900 mb-2">📊 周期性发现</h3>
              <p className="text-sm text-green-800">
                检测到的主要交易周期: <strong>{results.fourier.dominant_periods.slice(0, 3).map((p: any) => Math.round(p.period)).join('天, ')}天</strong>
              </p>
              {results.fourier.dominant_period_days && (
                <p className="text-xs text-green-700 mt-1">
                  当前主导周期: <strong>{Math.round(results.fourier.dominant_period_days)}天</strong> → 
                  建议DTE: {results.butterfly.dte}天
                </p>
              )}
            </div>
          </div>

          {/* ARIMA价格预测 */}
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <TrendingUp className="w-6 h-6 text-blue-600" />
              ARIMA 价格预测与95%置信区间
              {results.arima.model_order && (
                <span className="text-xs font-normal text-gray-500 ml-2">
                  (模型: ARIMA{JSON.stringify(results.arima.model_order)})
                </span>
              )}
            </h2>
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart data={results.chart_data.price_forecast}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} interval={Math.floor(results.chart_data.price_forecast.length / 10)} />
                <YAxis domain={['auto', 'auto']} />
                <Tooltip />
                <Legend />
                <Area type="monotone" dataKey="upper" stroke="none" fill="#fecaca" fillOpacity={0.3} name="95%上界" />
                <Area type="monotone" dataKey="lower" stroke="none" fill="#fecaca" fillOpacity={0.3} name="95%下界" />

                <Line type="monotone" dataKey="actual" stroke="#2563eb" strokeWidth={2} dot={false} name="实际价格" />
                <Line type="monotone" dataKey="forecast" stroke="#ef4444" strokeWidth={2} strokeDasharray="5 5" dot={{ r: 3 }} name="ARIMA预测" />

                <ReferenceLine y={results.butterfly.center_strike} stroke="#10b981" strokeDasharray="3 3" label="中心" />
                <ReferenceLine y={results.butterfly.lower_strike} stroke="#f59e0b" strokeDasharray="3 3" label="下翼" />
                <ReferenceLine y={results.butterfly.upper_strike} stroke="#f59e0b" strokeDasharray="3 3" label="上翼" />
                {results.butterfly.breakeven_lower && (
                  <>
                    <ReferenceLine y={results.butterfly.breakeven_lower} stroke="#dc2626" strokeDasharray="2 2" label="BEP-" />
                    <ReferenceLine y={results.butterfly.breakeven_upper} stroke="#dc2626" strokeDasharray="2 2" label="BEP+" />
                  </>
                )}
              </AreaChart>
            </ResponsiveContainer>

            <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
              <h3 className="font-semibold text-blue-900 mb-2">🎯 蝴蝶期权入场逻辑</h3>
              <ul className="text-sm text-blue-800 space-y-1">
                <li>• <strong>95%置信区间</strong>（粉色区域）：${results.lower_bound.toFixed(1)} - ${results.upper_bound.toFixed(1)}</li>
                <li>• <strong>预测中心价格</strong>：${results.forecast_price.toFixed(2)} （红色虚线）</li>
                <li>• <strong>蝴蝶中心行权价</strong>：${results.butterfly.center_strike} （绿色虚线）</li>
                {results.butterfly.breakeven_lower && (
                  <li>• <strong>盈亏平衡点</strong>：${results.butterfly.breakeven_lower.toFixed(2)} ~ ${results.butterfly.breakeven_upper.toFixed(2)} （红色虚线）</li>
                )}
                <li>• <strong>策略逻辑</strong>：置信区间窄 → 价格稳定 → 适合蝴蝶期权</li>
              </ul>
            </div>
          </div>

          {/* GARCH波动率预测 */}
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <Activity className="w-6 h-6 text-purple-600" />
              GARCH 波动率预测
              {results.garch.garch_params && (
                <span className="text-xs font-normal text-gray-500 ml-2">
                  (ω={results.garch.garch_params.omega.toFixed(4)}, 
                   α={results.garch.garch_params.alpha.toFixed(4)}, 
                   β={results.garch.garch_params.beta.toFixed(4)})
                </span>
              )}
            </h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={results.chart_data.volatility}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} interval={Math.floor(results.chart_data.volatility.length / 8)} />
                <YAxis domain={[0, 0.4]} tickFormatter={(val) => `${(val * 100).toFixed(0)}%`} />
                <Tooltip formatter={(val: any) => `${(val * 100).toFixed(1)}%`} />
                <Legend />
                <Line type="monotone" dataKey="realized" stroke="#8b5cf6" strokeWidth={2} dot={false} name="历史波动率" />
                <Line type="monotone" dataKey="predicted" stroke="#ec4899" strokeWidth={2} strokeDasharray="5 5" dot={{ r: 3 }} name="GARCH预测" />
                <ReferenceLine y={results.garch.current_iv} stroke="#f59e0b" strokeDasharray="3 3" label={`当前IV: ${(results.garch.current_iv * 100).toFixed(1)}%`} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* 蝴蝶期权详情 */}
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <DollarSign className="w-6 h-6 text-green-600" />
              蝴蝶期权构建方案
              <span className="text-xs font-normal text-gray-500 ml-2">
                (DTE: {results.butterfly.dte}天, r={results.butterfly.risk_free_rate.toFixed(2)}%)
              </span>
            </h2>

            {(() => {
              const { lower, center, upper, isIron } = getLegTypes(results.fourier.butterfly_type);
              return (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                  <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                    <p className="text-sm text-gray-600 mb-1">下翼 (买入 {lower})</p>
                    <p className="text-2xl font-bold text-gray-800">${results.butterfly.lower_strike.toFixed(0)}</p>
                    <p className="text-sm text-gray-600 mt-2">成本: ${results.butterfly.lower_cost.toFixed(2)}</p>
                    {results.butterfly.spreads && (
                      <p className="text-xs text-gray-500 mt-1">价差: {results.butterfly.spreads.lower.toFixed(1)}%</p>
                    )}
                  </div>

                  <div className={`p-4 rounded-lg border ${isIron ? 'bg-indigo-50 border-indigo-200' : 'bg-blue-50 border-blue-200'}`}>
                    <p className="text-sm text-gray-600 mb-1">中心 (卖出 {center})</p>
                    <p className="text-2xl font-bold text-gray-800">${results.butterfly.center_strike.toFixed(0)}</p>
                    
                    {/* Income Logic based on Strategy Type */}
                    <p className="text-sm text-gray-600 mt-2">收入: ${(results.butterfly.center_credit * 2).toFixed(2)}</p>

                    {results.butterfly.spreads && (
                      <p className="text-xs text-gray-500 mt-1">价差: {results.butterfly.spreads.center.toFixed(1)}%</p>
                    )}
                  </div>

                  <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                    <p className="text-sm text-gray-600 mb-1">上翼 (买入 {upper})</p>
                    <p className="text-2xl font-bold text-gray-800">${results.butterfly.upper_strike.toFixed(0)}</p>
                    <p className="text-sm text-gray-600 mt-2">成本: ${results.butterfly.upper_cost.toFixed(2)}</p>
                    {results.butterfly.spreads && (
                      <p className="text-xs text-gray-500 mt-1">价差: {results.butterfly.spreads.upper.toFixed(1)}%</p>
                    )}
                  </div>
                </div>
              );
            })()}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 bg-red-50 rounded-lg border border-red-200">
                <h3 className="font-semibold text-red-900 mb-2">最大风险</h3>
                <p className="text-3xl font-bold text-red-600">
                  {results.fourier.butterfly_type === 'IRON' 
                    ? `$${(results.butterfly.upper_strike - results.butterfly.center_strike - Math.abs(results.butterfly.net_debit)).toFixed(2)}` 
                    : `$${Math.abs(results.butterfly.net_debit).toFixed(2)}`
                  }
                </p>
                <div className="text-sm text-red-700 mt-1">
                  {results.fourier.butterfly_type === 'IRON' 
                    ? '翼宽 - 净收入' 
                    : '净权利金支出（初始成本）'
                  }
                </div>
                {results.butterfly.max_loss && (
                  <p className="text-xs text-red-600 mt-1">= 最大损失</p>
                )}
              </div>

              <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                <h3 className="font-semibold text-green-900 mb-2">最大收益</h3>
                <p className="text-3xl font-bold text-green-600">
                  {results.fourier.butterfly_type === 'IRON' 
                    ? `$${Math.abs(results.butterfly.net_debit).toFixed(2)}` 
                    : `$${results.butterfly.max_profit.toFixed(2)}`
                  }
                </p>
                
                <div className="text-sm text-green-700 mt-1 flex items-center gap-1">
                  {results.fourier.butterfly_type === 'IRON' ? (
                     <>净收入: ${Math.abs(results.butterfly.net_debit).toFixed(2)} (初始收入)</>
                  ) : (
                     <>价格在${results.butterfly.center_strike.toFixed(0)}时实现</>
                  )}
                </div>

                {results.butterfly.prob_profit && (
                  <p className="text-xs text-green-600 mt-1">预期盈利概率: {results.butterfly.prob_profit.toFixed(0)}%</p>
                )}
              </div>
            </div>

            {/* 交易建议区域 */}
            {results.trade_suggestion && (
              <div className="mt-4 p-4 bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg border-2 border-indigo-200">
                <h3 className="font-semibold text-indigo-900 mb-3 flex items-center gap-2">
                  <Target className="w-5 h-5" />
                  智能交易建议
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
                  <div className="bg-white/50 p-2 rounded">
                    <div className="text-xs text-gray-600">操作</div>
                    <div className={`font-bold ${getRecommendationStyle(results.trade_suggestion.action).color}`}>
                      {getRecommendationStyle(results.trade_suggestion.action).text}
                    </div>
                  </div>
                  <div className="bg-white/50 p-2 rounded">
                    <div className="text-xs text-gray-600">建议仓位</div>
                    <div className="font-bold text-gray-800">{results.trade_suggestion.position_size}</div>
                  </div>
                  <div className="bg-white/50 p-2 rounded">
                    <div className="text-xs text-gray-600">入场时机</div>
                    <div className="font-bold text-gray-800">
                      {results.trade_suggestion.entry_timing === 'IMMEDIATE' ? '立即' : '等待回调'}
                    </div>
                  </div>
                  <div className="bg-white/50 p-2 rounded">
                    <div className="text-xs text-gray-600">持有期</div>
                    <div className="font-bold text-gray-800">{results.trade_suggestion.hold_until}</div>
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                  <div className="bg-red-100/50 p-2 rounded">
                    <strong>止损:</strong> ${results.trade_suggestion.stop_loss}
                  </div>
                  <div className="bg-green-100/50 p-2 rounded">
                    <strong>止盈:</strong> ${results.trade_suggestion.take_profit}
                  </div>
                </div>
                {results.trade_suggestion.key_risks && results.trade_suggestion.key_risks.length > 0 && (
                  <div className="mt-3 bg-yellow-100/50 p-3 rounded">
                    <div className="text-xs font-semibold text-yellow-900 mb-1">⚠️ 关键风险:</div>
                    <ul className="text-xs text-yellow-800 space-y-0.5">
                      {results.trade_suggestion.key_risks.map((risk: string, idx: number) => (
                        <li key={idx}>• {risk}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}

            <div className="mt-4 p-4 bg-yellow-50 rounded-lg border border-yellow-200">
              <h3 className="font-semibold text-yellow-900 mb-2">⚠️ 风险管理规则</h3>
              <ul className="text-sm text-yellow-800 space-y-1">
                <li>• <strong>止损</strong>: 损失超过${(results.butterfly.net_debit * 0.5).toFixed(2)} (50%成本)</li>
                <li>• <strong>止盈</strong>: 盈利达到${(results.butterfly.max_profit * 0.7).toFixed(2)} (70%最大收益)</li>
                <li>• <strong>价格偏离</strong>: 超出${(results.butterfly.lower_strike - 3).toFixed(0)}-${(results.butterfly.upper_strike + 3).toFixed(0)}立即平仓</li>
                <li>• <strong>波动率飙升</strong>: IV上涨30%以上考虑退出</li>
                <li>• <strong>时间管理</strong>: 剩余7天到期时强制平仓</li>
                {results.butterfly.greeks && Math.abs(results.butterfly.greeks.delta) > 0.15 && (
                  <li className="text-red-600">• <strong>Delta风险</strong>: 当前Delta={results.butterfly.greeks.delta.toFixed(3)}，存在方向性风险</li>
                )}
              </ul>
            </div>
          </div>

          {/* 交易检查清单 */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <CheckCircle className="w-6 h-6 text-green-600" />
              交易检查清单
            </h2>

            <div className="space-y-3">
              {[
                { label: '价格稳定性良好 (波动<12%)', status: results.signals.price_stability },
                { label: '波动率被高估 (IV > GARCH预测)', status: results.signals.vol_mispricing },
                { label: '傅立叶趋势明确', status: results.signals.trend_clear },
                { label: '周期位置匹配策略', status: results.signals.cycle_aligned },
                ...(results.signals.delta_neutral ? [
                  { label: 'Delta中性 (|Δ| < 0.10)', status: results.signals.delta_neutral }
                ] : []),
                ...(results.signals.iv_high ? [
                  { label: 'IV在高位 (百分位 > 60%)', status: results.signals.iv_high }
                ] : [])
              ].map((item, idx) => (
                <div key={idx} className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                  {getSignalIcon(item.status)}
                  <span className={`text-sm ${item.status ? 'text-gray-700' : 'text-red-600'}`}>
                    {item.label}
                  </span>
                </div>
              ))}
            </div>

            <div className={`mt-4 p-4 rounded-lg border-2 ${
              Object.values(results.signals).filter(s => s).length >= Math.ceil(Object.keys(results.signals).length * 0.7)
                ? 'text-green-600 bg-green-50 border-green-200'
                : 'text-yellow-600 bg-yellow-50 border-yellow-200'
            }`}>
              <p className="font-bold text-lg mb-2">
                {Object.values(results.signals).filter(s => s).length >= Math.ceil(Object.keys(results.signals).length * 0.7)
                  ? `✅ 所有条件满足，建议入场 ${results.fourier.butterfly_type} Butterfly！`
                  : '⚠️ 条件未完全满足，建议等待或小仓位测试'}
              </p>
              <p className="text-sm">
                通过检查项: <strong>{Object.values(results.signals).filter(s => s).length}/{Object.keys(results.signals).length}</strong> |
                风险等级: <strong>{results.risk_level}</strong> |
                {results.score && (
                  <> 综合评分: <strong>{results.score.total}分</strong> | </>
                )}
                策略置信度: <strong>{results.score ? results.score.confidence_level : `${results.confidence}%`}</strong> |
                建议仓位: <strong>{
                  results.score && results.score.total > 75 ? '3-5%' :
                  results.score && results.score.total > 60 ? '2-3%' : '1-2%'
                }</strong>总资金
              </p>
            </div>
          </div>
        </>
      )}

      {!results && !error && (
        <div className="bg-white rounded-lg shadow-lg p-12 text-center">
          <Waves className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-600 text-lg">输入股票代码并点击"开始分析"</p>
          <p className="text-gray-500 text-sm mt-2">系统将自动判断使用 Call/Put/Iron Butterfly 策略</p>
          <p className="text-gray-500 text-sm mt-4">推荐标的: AAPL, MSFT, GOOGL, TSLA, SPY, QQQ</p>
          <div className="mt-6 p-4 bg-blue-50 rounded-lg text-left">
            <h3 className="font-bold text-blue-900 mb-2">🆕 v2.1 新增功能</h3>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>✅ 真实期权链IV + IV Skew精确建模</li>
              <li>✅ Black-Scholes精确定价（取代魔法数字）</li>
              <li>✅ 完整Greeks风险指标（Delta/Gamma/Vega/Theta）</li>
              <li>✅ 多因子综合评分系统（0-100分）</li>
              <li>✅ 智能交易建议（止损/止盈/仓位）</li>
              <li>✅ ARIMA自动参数选择</li>
              <li>✅ 傅立叶VWAP去趋势（真正的频域分析）</li>
            </ul>
          </div>
        </div>
      )}

      <div className="mt-6 p-4 bg-gray-100 rounded-lg text-center text-xs text-gray-600">
        <p>⚠️ 本工具仅供教育和研究使用，不构成投资建议。期权交易存在风险，请谨慎决策。</p>
        <p className="mt-1">数据来源: Yahoo Finance | 分析方法: FFT(VWAP) + ARIMA + GARCH + BS + Greeks</p>
        <p className="mt-1">版本: v2.1 | 后端: Python Flask | 前端: React + Recharts</p>
      </div>
    </div>
  );
};
export default ButterflyOptionAnalyzer;