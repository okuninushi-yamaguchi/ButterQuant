import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Area, AreaChart, BarChart, Bar } from 'recharts';
import { TrendingUp, Activity, DollarSign, AlertTriangle, CheckCircle, Play, Waves, TrendingDown, Award, Shield, Target, Info, Layout, List, BarChart2 } from 'lucide-react';
import { Helmet } from 'react-helmet-async';
import { useTranslation } from 'react-i18next';
import { config } from '../config';

// Use a typed constant for the custom element to bypass JSX.IntrinsicElements check
const TvMiniChart = 'tv-mini-chart' as any;

interface ButterflyOptionAnalyzerProps {
  initialTicker?: string;
}

const ButterflyOptionAnalyzer: React.FC<ButterflyOptionAnalyzerProps> = ({ initialTicker }) => {
  const { t } = useTranslation();
  const [ticker, setTicker] = useState<string>(initialTicker || '');
  const [analyzedTicker, setAnalyzedTicker] = useState<string>('');
  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'charts' | 'details'>('overview');

  // Auto-run if initialTicker is provided
  React.useEffect(() => {
    if (initialTicker) {
      setTicker(initialTicker);
      runAnalysis(initialTicker);
    }
  }, [initialTicker]);

  // Calculate Dynamic Title
  const pageTitle = (() => {
    if (results && ticker.toUpperCase() === analyzedTicker) {
      return t('analyzer.page_title', { ticker: analyzedTicker });
    } else {
      return t('analyzer.page_title_default');
    }
  })();

  const runAnalysis = async (tickerOverride?: string | any) => {
    // Ensure tickerOverride is a string (and not an event object)
    const targetTicker = typeof tickerOverride === 'string' ? tickerOverride : ticker;
    if (!targetTicker) return;

    setAnalyzing(true);
    setError(null);
    setResults(null);
    setActiveTab('overview');

    try {
      const response = await fetch(`${config.API_URL}/api/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ticker: targetTicker })
      });

      const data = await response.json();

      if (data.success) {
        setResults(data.data);
        setAnalyzedTicker(targetTicker.toUpperCase());
      } else {
        setError(data.error || t('common.error'));
      }
    } catch (err: any) {
      setError(t('analyzer.error_connect', { error: err.message }));
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
      'STRONG_BUY': { color: 'text-green-700', bg: 'bg-green-100', border: 'border-green-300', text: t('analyzer.score_card.rec.STRONG_BUY') },
      'BUY': { color: 'text-green-600', bg: 'bg-green-50', border: 'border-green-200', text: t('analyzer.score_card.rec.BUY') },
      'NEUTRAL': { color: 'text-yellow-600', bg: 'bg-yellow-50', border: 'border-yellow-200', text: t('analyzer.score_card.rec.NEUTRAL') },
      'AVOID': { color: 'text-red-600', bg: 'bg-red-50', border: 'border-red-200', text: t('analyzer.score_card.rec.AVOID') }
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
    <div className="w-full max-w-7xl mx-auto p-4 md:p-6 bg-gradient-to-br from-blue-50 to-indigo-50 min-h-screen">
      <Helmet>
        <title>{pageTitle}</title>
      </Helmet>

      {/* Header */}
      <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
        <div className="flex flex-col md:flex-row md:items-start justify-between gap-6 mb-6">
          <div className="flex-1">
            <h1 className="text-3xl font-bold text-gray-800 flex items-center gap-2">
              <Waves className="w-8 h-8 text-blue-600" />
              {t('analyzer.title')}
              <span className="text-sm font-normal text-blue-600 bg-blue-100 px-2 py-1 rounded ml-2">v2.1</span>
            </h1>
            <p className="text-gray-600 mt-2">{t('analyzer.subtitle')}</p>
            <div className="flex flex-wrap gap-2 mt-3">
              <span className="text-xs bg-purple-100 text-purple-700 px-2.5 py-1 rounded-full border border-purple-200">✨ {t('analyzer.features.iv_skew')}</span>
              <span className="text-xs bg-green-100 text-green-700 px-2.5 py-1 rounded-full border border-green-200">✨ {t('analyzer.features.bs_pricing')}</span>
              <span className="text-xs bg-blue-100 text-blue-700 px-2.5 py-1 rounded-full border border-blue-200">✨ {t('analyzer.features.greeks')}</span>
              <span className="text-xs bg-yellow-100 text-yellow-700 px-2.5 py-1 rounded-full border border-yellow-200">✨ {t('analyzer.features.score')}</span>
            </div>
          </div>

          <div className="w-full md:w-[350px] shrink-0 h-[150px] bg-gray-50 rounded-lg overflow-hidden border border-gray-200 shadow-inner">
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

        <div className="flex gap-4 items-end bg-gray-50 p-4 rounded-lg border border-gray-100">
          <div className="flex-1">
            <label className="block text-sm font-medium text-gray-700 mb-2">{t('dashboard.ticker_placeholder')}</label>
            <div className="relative">
              <input
                type="text"
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                className="w-full px-4 py-3 pl-10 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all shadow-sm"
                placeholder={t('analyzer.input_placeholder')}
                onKeyPress={(e) => e.key === 'Enter' && runAnalysis()}
              />
              <Target className="w-5 h-5 text-gray-400 absolute left-3 top-3.5" />
            </div>
          </div>
          <button
            onClick={() => runAnalysis()}
            disabled={analyzing || !ticker}
            className="px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center gap-2 transition-all shadow-md font-medium"
          >
            {analyzing ? (
              <>
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                {t('analyzer.analyzing')}
              </>
            ) : (
              <>
                <Play className="w-5 h-5 fill-current" />
                {t('analyzer.start_analysis')}
              </>
            )}
          </button>
        </div>

        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 flex items-center gap-3 animate-fade-in">
            <AlertTriangle className="w-5 h-5 shrink-0" />
            <div>
              <p className="font-semibold">{t('common.error')}</p>
              <p className="text-sm">{error}</p>
            </div>
          </div>
        )}
      </div>

      {results && (
        <div className="space-y-6">
          {/* Use Tab approach for cleaner layout */}
          <div className="flex space-x-1 bg-white p-1 rounded-xl shadow-sm border border-gray-200">
            <button
              onClick={() => setActiveTab('overview')}
              className={`flex-1 flex items-center justify-center gap-2 py-2.5 text-sm font-medium rounded-lg transition-all ${activeTab === 'overview' ? 'bg-indigo-50 text-indigo-700 shadow-sm' : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'}`}
            >
              <Layout className="w-4 h-4" />
              Overview
            </button>
            <button
              onClick={() => setActiveTab('charts')}
              className={`flex-1 flex items-center justify-center gap-2 py-2.5 text-sm font-medium rounded-lg transition-all ${activeTab === 'charts' ? 'bg-indigo-50 text-indigo-700 shadow-sm' : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'}`}
            >
              <BarChart2 className="w-4 h-4" />
              Deep Dive & Charts
            </button>
            <button
              onClick={() => setActiveTab('details')}
              className={`flex-1 flex items-center justify-center gap-2 py-2.5 text-sm font-medium rounded-lg transition-all ${activeTab === 'details' ? 'bg-indigo-50 text-indigo-700 shadow-sm' : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'}`}
            >
              <List className="w-4 h-4" />
              Details & Greeks
            </button>
          </div>

          <div className="animate-fade-in">
            {activeTab === 'overview' && (
              <>
                {/* Score Card */}
                {results.score && (
                  <div className="bg-gradient-to-r from-purple-600 to-indigo-700 rounded-xl shadow-xl p-6 mb-6 text-white relative overflow-hidden">
                    <div className="absolute top-0 right-0 p-3 opacity-10">
                      <Award className="w-64 h-64" />
                    </div>
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 relative z-10">
                      <div>
                        <div className="flex items-center gap-3 mb-2">
                          <Award className="w-10 h-10 text-yellow-300" />
                          <h2 className="text-3xl font-bold">{t('analyzer.score_card.title')}</h2>
                        </div>
                        <p className="text-purple-100 text-sm max-w-lg">{t('analyzer.score_card.subtitle')}</p>

                        <div className="mt-4 flex items-center gap-3">
                          <div className={`px-4 py-2 rounded-lg font-black text-lg bg-white text-indigo-900 shadow-lg border-2 border-indigo-200 flex items-center gap-2`}>
                            {getTrendIcon(results.fourier.trend_direction)}
                            {results.fourier.butterfly_type} BUTTERFLY
                          </div>
                        </div>

                        <div className="mt-6 flex flex-wrap gap-3">
                          <div className={`px-4 py-2 rounded-lg font-bold bg-white/20 backdrop-blur-sm border border-white/30`}>
                            {getRecommendationStyle(results.score.recommendation).text}
                          </div>
                          <div className="px-4 py-2 rounded-lg bg-white/10 backdrop-blur-sm border border-white/20">
                            {t('analyzer.score_card.confidence')}: <span className="font-bold">{results.score.confidence_level}</span>
                          </div>
                        </div>
                      </div>
                      <div className="text-right bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20 shadow-inner">
                        <div className="text-7xl font-black tracking-tighter">{results.score.total}</div>
                        <div className="text-sm font-medium text-purple-200 tracking-wider">OUT OF 100</div>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8 relative z-10">
                      {[
                        { l: t('analyzer.score_card.price_match'), v: results.score.components.price_match, w: '35%' },
                        { l: t('analyzer.score_card.vol_mispricing'), v: results.score.components.vol_mispricing, w: '30%' },
                        { l: t('analyzer.score_card.stability'), v: results.score.components.stability, w: '20%' },
                        { l: t('analyzer.score_card.cycle_align'), v: results.score.components.fourier_align, w: '15%' },
                      ].map((item, i) => (
                        <div key={i} className="bg-black/20 backdrop-blur-sm rounded-lg p-3 border border-white/10 transition-transform hover:scale-105">
                          <div className="text-xs text-purple-200 mb-1 font-medium uppercase tracking-wider">{item.l}</div>
                          <div className="text-2xl font-bold">{item.v}</div>
                          <div className="text-[10px] text-purple-300 mt-1">Weight: {item.w}</div>
                        </div>
                      ))}
                    </div>

                    {results.score.components.delta_penalty > 0 && (
                      <div className="mt-4 bg-yellow-500/20 border border-yellow-300/30 rounded p-2 text-sm flex items-center gap-2 relative z-10">
                        <AlertTriangle className="w-4 h-4 text-yellow-300" />
                        <span>{t('analyzer.score_card.delta_penalty')}: -{results.score.components.delta_penalty}</span>
                      </div>
                    )}
                  </div>
                )}

                {/* Core Metrics */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                  <div className="bg-white rounded-xl shadow-sm p-5 border border-gray-100 hover:shadow-md transition-shadow">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-500">{t('analyzer.metrics.price_stability')}</span>
                      {getSignalIcon(results.signals.price_stability)}
                    </div>
                    <p className="text-3xl font-bold text-gray-800">{results.price_stability}%</p>
                    <p className="text-xs text-gray-400 mt-1">{t('analyzer.metrics.stability_desc')}</p>
                  </div>

                  <div className="bg-white rounded-xl shadow-sm p-5 border border-gray-100 hover:shadow-md transition-shadow">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-500">{t('analyzer.metrics.vol_mispricing')}</span>
                      {getSignalIcon(results.signals.vol_mispricing)}
                    </div>
                    <p className={`text-3xl font-bold ${(results.garch.vol_mispricing) > 0 ? 'text-green-600' : 'text-gray-800'}`}>
                      {results.garch.vol_mispricing > 0 ? '+' : ''}{results.garch.vol_mispricing.toFixed(1)}%
                    </p>
                    <p className="text-xs text-gray-400 mt-1">{t('analyzer.metrics.vol_desc')}</p>
                  </div>

                  <div className="bg-white rounded-xl shadow-sm p-5 border border-gray-100 hover:shadow-md transition-shadow">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-500">{t('analyzer.metrics.profit_ratio')}</span>
                      <DollarSign className="w-5 h-5 text-green-500" />
                    </div>
                    <p className="text-3xl font-bold text-gray-800">{results.butterfly.profit_ratio.toFixed(1)}x</p>
                    <p className="text-xs text-gray-400 mt-1">{t('analyzer.metrics.profit_desc')}</p>
                  </div>

                  <div className="bg-white rounded-xl shadow-sm p-5 border border-gray-100 hover:shadow-md transition-shadow">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-500">{t('analyzer.metrics.risk_level')}</span>
                      <Shield className={`w-5 h-5 ${results.risk_level === 'LOW' ? 'text-green-500' : 'text-yellow-500'}`} />
                    </div>
                    <p className={`text-3xl font-bold ${results.risk_level === 'LOW' ? 'text-green-600' : results.risk_level === 'MEDIUM' ? 'text-yellow-600' : 'text-red-600'}`}>
                      {results.risk_level}
                    </p>
                    <div className="h-1.5 w-full bg-gray-100 rounded-full mt-2 overflow-hidden">
                      <div className={`h-full rounded-full ${results.risk_level === 'LOW' ? 'bg-green-500 w-1/3' : results.risk_level === 'MEDIUM' ? 'bg-yellow-500 w-2/3' : 'bg-red-500 w-full'}`}></div>
                    </div>
                  </div>
                </div>

                {/* Suggestions */}
                {results.trade_suggestion && (
                  <div className="p-6 bg-white rounded-xl shadow-lg border-l-4 border-indigo-500 mb-6">
                    <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                      <Target className="w-5 h-5 text-indigo-600" />
                      {t('analyzer.suggestion.title')}
                    </h3>
                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
                      <div className="bg-gray-50 p-3 rounded-lg border border-gray-100">
                        <div className="text-xs text-gray-500 uppercase font-semibold mb-1">{t('analyzer.suggestion.action')}</div>
                        <div className={`text-xl font-bold ${getRecommendationStyle(results.trade_suggestion.action).color}`}>
                          {getRecommendationStyle(results.trade_suggestion.action).text}
                        </div>
                      </div>
                      <div className="bg-gray-50 p-3 rounded-lg border border-gray-100">
                        <div className="text-xs text-gray-500 uppercase font-semibold mb-1">{t('analyzer.suggestion.position')}</div>
                        <div className="text-xl font-bold text-gray-800">{results.trade_suggestion.position_size}</div>
                      </div>
                      <div className="bg-gray-50 p-3 rounded-lg border border-gray-100">
                        <div className="text-xs text-gray-500 uppercase font-semibold mb-1">{t('analyzer.suggestion.entry')}</div>
                        <div className="text-lg font-bold text-gray-800">
                          {results.trade_suggestion.entry_timing === 'IMMEDIATE' ? t('analyzer.suggestion.immediate') : t('analyzer.suggestion.wait')}
                        </div>
                      </div>
                      <div className="bg-gray-50 p-3 rounded-lg border border-gray-100">
                        <div className="text-xs text-gray-500 uppercase font-semibold mb-1">{t('analyzer.suggestion.hold')}</div>
                        <div className="text-lg font-bold text-gray-800">{results.trade_suggestion.hold_until}</div>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="flex items-center justify-between p-3 bg-red-50 rounded-lg border border-red-100">
                        <span className="font-semibold text-red-900 text-sm">{t('analyzer.suggestion.stop_loss')}</span>
                        <span className="font-mono font-bold text-red-700">${results.trade_suggestion.stop_loss}</span>
                      </div>
                      <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg border border-green-100">
                        <span className="font-semibold text-green-900 text-sm">{t('analyzer.suggestion.take_profit')}</span>
                        <span className="font-mono font-bold text-green-700">${results.trade_suggestion.take_profit}</span>
                      </div>
                    </div>

                    {results.trade_suggestion.key_risks && results.trade_suggestion.key_risks.length > 0 && (
                      <div className="mt-4 pt-4 border-t border-gray-100">
                        <div className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-2">
                          <AlertTriangle className="w-4 h-4 text-amber-500" />
                          {t('analyzer.suggestion.key_risks')}
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {results.trade_suggestion.key_risks.map((risk: string, idx: number) => (
                            <span key={idx} className="text-xs bg-amber-50 text-amber-800 px-2 py-1 rounded border border-amber-100">• {t(`analyzer.suggestion.${risk}`)}</span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Strategy Snapshot */}
                <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                      <DollarSign className="w-6 h-6 text-green-600" />
                      Strategy Structure
                    </h2>
                    <span className="text-sm px-3 py-1 bg-gray-100 rounded-full text-gray-600 font-medium">
                      {results.butterfly.dte} Days DTE
                    </span>
                  </div>

                  {(() => {
                    const { lower, center, upper, isIron } = getLegTypes(results.fourier.butterfly_type);
                    return (
                      <div className="flex flex-col md:flex-row gap-0 items-stretch rounded-lg overflow-hidden border border-gray-200">
                        <div className="flex-1 p-4 bg-green-50/50 border-b md:border-b-0 md:border-r border-green-100 text-center">
                          <div className="text-xs text-gray-500 uppercase font-bold mb-1">{t('analyzer.strategy.lower_wing', { type: lower })}</div>
                          <div className="text-2xl font-bold text-gray-800">${results.butterfly.lower_strike.toFixed(0)}</div>
                          <div className="text-xs text-red-500 mt-1 block">Cost: ${results.butterfly.lower_cost.toFixed(2)}</div>
                        </div>

                        <div className={`flex-1 p-4 text-center border-b md:border-b-0 md:border-r border-gray-200 relative ${isIron ? 'bg-indigo-50/50' : 'bg-blue-50/50'}`}>
                          <div className="absolute top-0 left-1/2 -translate-x-1/2 bg-blue-600 text-white text-[10px] px-2 py-0.5 rounded-b-md shadow-sm">Target</div>
                          <div className="text-xs text-gray-500 uppercase font-bold mb-1">{t('analyzer.strategy.center_wing', { type: center })}</div>
                          <div className="text-3xl font-black text-gray-800">${results.butterfly.center_strike.toFixed(0)}</div>
                          <div className="text-xs text-green-600 mt-1 font-bold">Income: ${(results.butterfly.center_credit * 2).toFixed(2)}</div>
                        </div>

                        <div className="flex-1 p-4 bg-green-50/50 text-center">
                          <div className="text-xs text-gray-500 uppercase font-bold mb-1">{t('analyzer.strategy.upper_wing', { type: upper })}</div>
                          <div className="text-2xl font-bold text-gray-800">${results.butterfly.upper_strike.toFixed(0)}</div>
                          <div className="text-xs text-red-500 mt-1 block">Cost: ${results.butterfly.upper_cost.toFixed(2)}</div>
                        </div>
                      </div>
                    );
                  })()}

                  <div className="grid grid-cols-2 gap-4 mt-4">
                    <div className="p-3 rounded border border-red-100 bg-red-50/30 flex justify-between items-center">
                      <span className="text-sm text-gray-600">{t('analyzer.strategy.max_risk')}</span>
                      <span className="font-bold text-red-600">
                        {results.fourier.butterfly_type === 'IRON'
                          ? `$${(results.butterfly.upper_strike - results.butterfly.center_strike - Math.abs(results.butterfly.net_debit)).toFixed(2)}`
                          : `$${Math.abs(results.butterfly.net_debit).toFixed(2)}`
                        }
                      </span>
                    </div>
                    <div className="p-3 rounded border border-green-100 bg-green-50/30 flex justify-between items-center">
                      <span className="text-sm text-gray-600">{t('analyzer.strategy.max_profit')}</span>
                      <span className="font-bold text-green-600">
                        {results.fourier.butterfly_type === 'IRON'
                          ? `$${Math.abs(results.butterfly.net_debit).toFixed(2)}`
                          : `$${results.butterfly.max_profit.toFixed(2)}`
                        }
                      </span>
                    </div>
                  </div>
                </div>
              </>
            )}

            {activeTab === 'charts' && (
              <div className="space-y-6">
                {/* Fourier Chart */}
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                    <Waves className="w-6 h-6 text-purple-600" />
                    {t('analyzer.fourier.chart_title')}
                  </h2>
                  <div className="h-[400px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={results.chart_data.fourier}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                        <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#6b7280' }} interval={Math.floor(results.chart_data.fourier.length / 10)} />
                        <YAxis domain={['auto', 'auto']} tick={{ fontSize: 11, fill: '#6b7280' }} />
                        <Tooltip
                          contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                        />
                        <Legend verticalAlign="top" height={36} />

                        <Line type="monotone" dataKey="actual" stroke="#3b82f6" strokeWidth={2} dot={false} name="Actual Price" />
                        <Line type="monotone" dataKey="lowFreq" stroke="#ef4444" strokeWidth={3} dot={false} name="Low Freq Trend" />
                        <Line type="monotone" dataKey="midFreq" stroke="#10b981" strokeWidth={2} dot={false} strokeDasharray="5 5" name="Cycle Component" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Spectrum Chart */}
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                    <Activity className="w-6 h-6 text-green-600" />
                    {t('analyzer.fourier.spectrum_title')}
                  </h2>
                  <div className="flex flex-col md:flex-row gap-6">
                    <div className="flex-1 h-[300px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={results.chart_data.spectrum}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                          <XAxis dataKey="period" tick={{ fontSize: 12 }} label={{ value: 'Days', position: 'insideBottom', offset: -5 }} />
                          <YAxis />
                          <Tooltip cursor={{ fill: '#f0fdf4' }} />
                          <Bar dataKey="power" fill="#10b981" radius={[4, 4, 0, 0]} name="Cycle Power" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="md:w-1/3 p-4 bg-green-50 rounded-lg border border-green-200 h-fit">
                      <h3 className="font-semibold text-green-900 mb-3">Dominant Cycles</h3>
                      <div className="space-y-3">
                        {results.fourier.dominant_periods.slice(0, 3).map((p: any, i: number) => (
                          <div key={i} className="flex justify-between items-center bg-white p-2 rounded border border-green-100">
                            <span className="font-bold text-gray-700">{Math.round(p.period)} Days</span>
                            <span className="text-xs text-green-600 font-medium">Power: {p.power_pct.toFixed(1)}%</span>
                          </div>
                        ))}
                      </div>
                      <div className="mt-4 pt-3 border-t border-green-200">
                        <p className="text-xs text-green-800">
                          The current dominant cycle suggests a trade duration of <strong>{results.butterfly.dte} days</strong> would be optimal.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* ARIMA Forecast */}
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <div className="flex justify-between items-center mb-4">
                    <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                      <TrendingUp className="w-6 h-6 text-blue-600" />
                      {t('analyzer.forecast.title')}
                    </h2>
                    {results.arima.model_order && (
                      <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded font-mono">
                        ARIMA{JSON.stringify(results.arima.model_order)}
                      </span>
                    )}
                  </div>

                  <div className="h-[400px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={results.chart_data.price_forecast}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                        <XAxis dataKey="date" tick={{ fontSize: 11 }} interval={Math.floor(results.chart_data.price_forecast.length / 10)} />
                        <YAxis domain={['auto', 'auto']} />
                        <Tooltip />
                        <Legend verticalAlign="top" />
                        <defs>
                          <linearGradient id="splitColor" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#fecaca" stopOpacity={0.8} />
                            <stop offset="95%" stopColor="#fecaca" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <Area type="monotone" dataKey="upper" stroke="none" fill="#fecaca" fillOpacity={0.4} name="Confidence Interval" />
                        <Area type="monotone" dataKey="lower" stroke="none" fill="#fff" fillOpacity={1} />

                        <Line type="monotone" dataKey="actual" stroke="#2563eb" strokeWidth={2} dot={false} name="Actual" />
                        <Line type="monotone" dataKey="forecast" stroke="#dc2626" strokeWidth={2} strokeDasharray="5 5" dot={{ r: 3 }} name="Forecast" />

                        <ReferenceLine y={results.butterfly.center_strike} stroke="#10b981" strokeDasharray="3 3" label={{ value: 'Center', position: 'right', fill: '#10b981', fontSize: 10 }} />
                        <ReferenceLine y={results.butterfly.lower_strike} stroke="#f59e0b" strokeDasharray="3 3" label={{ value: 'Lower', position: 'right', fill: '#f59e0b', fontSize: 10 }} />
                        <ReferenceLine y={results.butterfly.upper_strike} stroke="#f59e0b" strokeDasharray="3 3" label={{ value: 'Upper', position: 'right', fill: '#f59e0b', fontSize: 10 }} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* GARCH */}
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                    <Activity className="w-6 h-6 text-pink-600" />
                    {t('analyzer.vol_forecast.title')}
                  </h2>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={results.chart_data.volatility}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                        <XAxis dataKey="date" tick={{ fontSize: 11 }} interval={Math.floor(results.chart_data.volatility.length / 8)} />
                        <YAxis domain={[0, 0.4]} tickFormatter={(val) => `${(val * 100).toFixed(0)}%`} />
                        <Tooltip formatter={(val: any) => `${(val * 100).toFixed(1)}%`} />
                        <Legend />
                        <Line type="monotone" dataKey="realized" stroke="#8b5cf6" strokeWidth={2} dot={false} name="Realized Vol" />
                        <Line type="monotone" dataKey="predicted" stroke="#ec4899" strokeWidth={2} strokeDasharray="5 5" dot={{ r: 3 }} name="GARCH Forecast" />
                        <ReferenceLine y={results.garch.current_iv} stroke="#f59e0b" strokeDasharray="3 3" label={`${t('analyzer.vol_forecast.current_iv')}: ${(results.garch.current_iv * 100).toFixed(1)}%`} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'details' && (
              <div className="space-y-6">
                {/* Greeks Panel */}
                {results.butterfly.greeks && (
                  <div className="bg-white rounded-xl shadow-lg p-6">
                    <h2 className="text-xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                      <Shield className="w-6 h-6 text-indigo-600" />
                      {t('analyzer.greeks.title')}
                    </h2>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                      <div className="p-4 bg-blue-50 rounded-xl border border-blue-100 flex flex-col items-center text-center hover:scale-105 transition-transform">
                        <div className="text-xs text-gray-500 font-bold uppercase tracking-wider mb-2">{t('analyzer.greeks.delta')}</div>
                        <div className={`text-3xl font-black mb-1 ${Math.abs(results.butterfly.greeks.delta) < 0.10 ? 'text-green-600' : 'text-yellow-600'}`}>
                          {results.butterfly.greeks.delta.toFixed(4)}
                        </div>
                        <div className="text-xs px-2 py-1 bg-white rounded-full border border-blue-100 text-gray-600">
                          {Math.abs(results.butterfly.greeks.delta) < 0.10 ? t('analyzer.greeks.neutral') : t('analyzer.greeks.directional')}
                        </div>
                      </div>

                      <div className="p-4 bg-purple-50 rounded-xl border border-purple-100 flex flex-col items-center text-center hover:scale-105 transition-transform">
                        <div className="text-xs text-gray-500 font-bold uppercase tracking-wider mb-2">{t('analyzer.greeks.gamma')}</div>
                        <div className="text-3xl font-black text-purple-700 mb-1">
                          {results.butterfly.greeks.gamma.toFixed(4)}
                        </div>
                        <div className="text-xs px-2 py-1 bg-white rounded-full border border-purple-100 text-gray-600">
                          Stability
                        </div>
                      </div>

                      <div className="p-4 bg-pink-50 rounded-xl border border-pink-100 flex flex-col items-center text-center hover:scale-105 transition-transform">
                        <div className="text-xs text-gray-500 font-bold uppercase tracking-wider mb-2">{t('analyzer.greeks.vega')}</div>
                        <div className={`text-3xl font-black mb-1 ${results.butterfly.greeks.vega < 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {results.butterfly.greeks.vega.toFixed(4)}
                        </div>
                        <div className="text-xs px-2 py-1 bg-white rounded-full border border-pink-100 text-gray-600">
                          {results.butterfly.greeks.vega < 0 ? t('analyzer.greeks.short_vol') : t('analyzer.greeks.long_vol')}
                        </div>
                      </div>

                      <div className="p-4 bg-green-50 rounded-xl border border-green-100 flex flex-col items-center text-center hover:scale-105 transition-transform">
                        <div className="text-xs text-gray-500 font-bold uppercase tracking-wider mb-2">{t('analyzer.greeks.theta')}</div>
                        <div className={`text-3xl font-black mb-1 ${results.butterfly.greeks.theta > 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {results.butterfly.greeks.theta.toFixed(4)}
                        </div>
                        <div className="text-xs px-2 py-1 bg-white rounded-full border border-green-100 text-gray-600">
                          Daily Decay
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* IV Skew Data */}
                {results.garch.iv_skew && (
                  <div className="bg-white rounded-xl shadow-lg p-6">
                    <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                      <Activity className="w-6 h-6 text-purple-600" />
                      {t('analyzer.iv_skew.title')}
                    </h2>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                      <div className="rounded-xl border border-gray-100 overflow-hidden">
                        <div className="bg-gray-50 p-3 border-b border-gray-100 text-center font-semibold text-gray-700">OTM Put (95%)</div>
                        <div className="p-6 text-center">
                          <div className="text-2xl font-bold text-gray-800">{(results.garch.iv_skew.otm_put * 100).toFixed(1)}%</div>
                          <div className="text-xs text-red-500 mt-2 font-medium bg-red-50 inline-block px-2 py-1 rounded">
                            SMA {results.garch.iv_skew.skew_put > 0 ? '+' : ''}{results.garch.iv_skew.skew_put.toFixed(1)}%
                          </div>
                        </div>
                      </div>

                      <div className="rounded-xl border-2 border-indigo-100 overflow-hidden relative shadow-sm">
                        <div className="absolute top-0 inset-x-0 h-1 bg-indigo-500"></div>
                        <div className="bg-indigo-50 p-3 border-b border-indigo-100 text-center font-bold text-indigo-900">ATM (100%)</div>
                        <div className="p-6 text-center bg-indigo-50/30">
                          <div className="text-3xl font-black text-indigo-600">{(results.garch.iv_skew.atm * 100).toFixed(1)}%</div>
                          <div className="text-xs text-indigo-400 mt-2">Benchmark IV</div>
                        </div>
                      </div>

                      <div className="rounded-xl border border-gray-100 overflow-hidden">
                        <div className="bg-gray-50 p-3 border-b border-gray-100 text-center font-semibold text-gray-700">OTM Call (105%)</div>
                        <div className="p-6 text-center">
                          <div className="text-2xl font-bold text-gray-800">{(results.garch.iv_skew.otm_call * 100).toFixed(1)}%</div>
                          <div className="text-xs text-green-500 mt-2 font-medium bg-green-50 inline-block px-2 py-1 rounded">
                            SMA {results.garch.iv_skew.skew_call > 0 ? '+' : ''}{results.garch.iv_skew.skew_call.toFixed(1)}%
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Checklist */}
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                    <CheckCircle className="w-6 h-6 text-green-600" />
                    {t('analyzer.checklist.title')}
                  </h2>

                  <div className="space-y-3">
                    {[
                      { label: t('analyzer.checklist.stability'), status: results.signals.price_stability },
                      { label: t('analyzer.checklist.vol'), status: results.signals.vol_mispricing },
                      { label: t('analyzer.checklist.trend'), status: results.signals.trend_clear },
                      { label: t('analyzer.checklist.cycle'), status: results.signals.cycle_aligned },
                      ...(results.signals.delta_neutral ? [
                        { label: t('analyzer.checklist.delta'), status: results.signals.delta_neutral }
                      ] : []),
                      ...(results.signals.iv_high ? [
                        { label: t('analyzer.checklist.iv'), status: results.signals.iv_high }
                      ] : [])
                    ].map((item, idx) => (
                      <div key={idx} className="flex items-center gap-3 p-4 bg-gray-50 rounded-xl border border-gray-100">
                        {getSignalIcon(item.status)}
                        <span className={`font-medium ${item.status ? 'text-gray-700' : 'text-gray-400'}`}>
                          {item.label}
                        </span>
                      </div>
                    ))}
                  </div>

                  <div className={`mt-6 p-5 rounded-xl border-2 ${Object.values(results.signals).filter(s => s).length >= Math.ceil(Object.keys(results.signals).length * 0.7)
                    ? 'text-green-800 bg-green-50 border-green-200'
                    : 'text-yellow-800 bg-yellow-50 border-yellow-200'
                    }`}>
                    <p className="font-bold text-lg mb-2 flex items-center gap-2">
                      {Object.values(results.signals).filter(s => s).length >= Math.ceil(Object.keys(results.signals).length * 0.7)
                        ? <><CheckCircle className="w-5 h-5" /> {t('analyzer.checklist.success_msg', { type: results.fourier.butterfly_type })}</>
                        : <><AlertTriangle className="w-5 h-5" /> {t('analyzer.checklist.fail_msg')}</>}
                    </p>
                    <p className="text-sm opacity-80 pl-7">
                      Pass rate: {Object.values(results.signals).filter(s => s).length}/{Object.keys(results.signals).length}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Empty State */}
      {!results && !error && (
        <div className="bg-white rounded-xl shadow-lg p-16 text-center max-w-2xl mx-auto mt-12 border border-dashed border-gray-300">
          <div className="bg-blue-50 w-24 h-24 rounded-full flex items-center justify-center mx-auto mb-6">
            <Waves className="w-12 h-12 text-blue-500" />
          </div>
          <h2 className="text-2xl font-bold text-gray-800 mb-2">{t('analyzer.empty.text1')}</h2>
          <p className="text-gray-500 mb-8">{t('analyzer.empty.text2')}</p>

          <div className="grid grid-cols-2 gap-3 text-left max-w-sm mx-auto">
            <div className="px-3 py-2 bg-gray-50 rounded text-sm text-gray-600 font-medium">AAPL</div>
            <div className="px-3 py-2 bg-gray-50 rounded text-sm text-gray-600 font-medium">MSFT</div>
            <div className="px-3 py-2 bg-gray-50 rounded text-sm text-gray-600 font-medium">TSLA</div>
            <div className="px-3 py-2 bg-gray-50 rounded text-sm text-gray-600 font-medium">SPY</div>
          </div>
        </div>
      )}

      <div className="mt-12 text-center">
        <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/50 rounded-full text-xs text-gray-500 border border-gray-100">
          <span>Powered by ButterQuant Engine</span>
          <span className="w-1 h-1 bg-gray-300 rounded-full"></span>
          <span>v2.1.0</span>
        </div>
      </div>
    </div>
  );
};
export default ButterflyOptionAnalyzer;