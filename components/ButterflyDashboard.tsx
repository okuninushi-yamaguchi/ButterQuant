import React, { useState, useEffect } from 'react';
import { RefreshCw, Plus, X, Activity, AlertCircle } from 'lucide-react';

interface StrategyCardProps {
  data: any;
  loading?: boolean;
}

const StrategyCard: React.FC<StrategyCardProps> = ({ data, loading }) => {
  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-4 h-64 flex flex-col items-center justify-center animate-pulse border border-gray-200">
        <div className="h-6 w-20 bg-gray-200 rounded mb-4"></div>
        <div className="h-4 w-32 bg-gray-200 rounded mb-2"></div>
        <div className="h-4 w-24 bg-gray-200 rounded"></div>
      </div>
    );
  }

  if (!data) return null;

  const { ticker, fourier, butterfly } = data;
  const type = fourier.butterfly_type; // 'CALL' | 'PUT' | 'IRON'

  // Determine Leg Types
  const isIron = type === 'IRON';

  // Leg Type Definitions based on Strategy
  // IRON: Lower=Put, Center=Straddle(Call+Put), Upper=Call
  // CALL: Lower=Call, Center=2 Calls, Upper=Call
  // PUT: Lower=Put, Center=2 Puts, Upper=Put

  const upperLegType = isIron ? 'Call' : (type === 'CALL' ? 'Call' : 'Put');
  const lowerLegType = isIron ? 'Put' : (type === 'CALL' ? 'Call' : 'Put');
  const centerLegLabel = isIron ? 'Call & Put' : (type === 'CALL' ? '2 Calls' : '2 Puts');

  // Color logic
  let colorStyles = {
    wrapper: 'bg-white border-gray-200',
    text: 'text-gray-800',
    highlight: 'text-blue-600',
    boxBorder: 'border-blue-600',
    label: 'text-gray-500',
    cost: 'text-red-600', // Expense
    income: 'text-green-600' // Income
  };

  if (type === 'CALL') {
    colorStyles = {
      ...colorStyles,
      wrapper: 'bg-green-50 border-green-200',
      text: 'text-green-900',
      highlight: 'text-green-600',
      boxBorder: 'border-green-600',
      label: 'text-green-700'
    };
  } else if (type === 'PUT') {
    colorStyles = {
      ...colorStyles,
      wrapper: 'bg-red-50 border-red-200',
      text: 'text-red-900',
      highlight: 'text-red-600',
      boxBorder: 'border-red-600',
      label: 'text-red-700'
    };
  } else {
    // IRON - Blue Theme
    colorStyles = {
      ...colorStyles,
      wrapper: 'bg-blue-50 border-blue-200',
      text: 'text-blue-900',
      highlight: 'text-blue-600',
      boxBorder: 'border-blue-600',
      label: 'text-blue-700',
      cost: 'text-red-600',
      income: 'text-green-600'
    };
  }

  return (
    <div className={`rounded-lg shadow-lg p-5 border-2 relative overflow-hidden transition-transform hover:scale-[1.02] ${colorStyles.wrapper}`}>
      <div className="flex justify-between h-full">
        {/* Left Column: Ticker & Type */}
        <div className="flex flex-col justify-between w-5/12 pr-2">
          <div>
            <h3 className={`text-3xl font-black tracking-tight ${colorStyles.text}`}>{ticker}</h3>
            <div className={`mt-2 text-sm font-bold ${colorStyles.highlight}`}>
              {type} BUTTERFLY
            </div>
          </div>

          <div className={`border p-2 text-center rounded ${colorStyles.boxBorder} bg-opacity-10 bg-white backdrop-blur-sm`}>
            <div className={`text-xs ${colorStyles.text} opacity-80`}>
              {isIron ? 'Net Income (Credit)' : 'Net Cost (Debit)'}
            </div>
            <div className={`text-xl font-bold ${colorStyles.text}`}>
              ${Math.abs(butterfly.net_debit).toFixed(2)}
            </div>
          </div>
        </div>

        {/* Right Column: Structure (Butterfly Shape) */}
        <div className="flex flex-col justify-between w-7/12 pl-2 text-right space-y-2 relative font-mono">

          {/* Upper Wing */}
          <div className="relative z-10">
            <div className={`text-[10px] font-bold uppercase ${colorStyles.label} mb-0.5`}>
              上翼 (买入 {upperLegType})
            </div>
            <div className={`text-lg font-bold ${colorStyles.text}`}>${butterfly.upper_strike}</div>
            <div className={`text-[10px] ${colorStyles.cost}`}>
              支出 ${butterfly.upper_cost.toFixed(2)}
            </div>
          </div>

          {/* Center */}
          <div className="relative z-10 my-1 py-1 border-t border-b border-dashed border-gray-500/30">
            {isIron ? (
              // Iron Butterfly Center: Split into Call and Put
              <>
                <div className={`text-[10px] font-bold uppercase ${colorStyles.highlight} mb-0.5`}>
                  中心 (卖出 Straddle)
                </div>
                <div className={`text-xl font-black ${colorStyles.highlight} mb-1`}>${butterfly.center_strike}</div>
                <div className="flex flex-col gap-0.5">
                  <div className={`text-[10px] ${colorStyles.income}`}>
                    卖Call: +${butterfly.center_credit.toFixed(2)}
                  </div>
                  <div className={`text-[10px] ${colorStyles.income}`}>
                    卖Put: +${butterfly.center_credit.toFixed(2)}
                  </div>
                </div>
              </>
            ) : (
              // Standard Butterfly Center
              <>
                <div className={`text-[10px] font-bold uppercase ${colorStyles.highlight} mb-0.5`}>
                  中心 (卖出 {centerLegLabel})
                </div>
                <div className={`text-xl font-black ${colorStyles.highlight}`}>${butterfly.center_strike}</div>
                <div className={`text-[10px] ${colorStyles.income}`}>
                  收入 +${(butterfly.center_credit * 2).toFixed(2)}
                </div>
              </>
            )}
          </div>

          {/* Lower Wing */}
          <div className="relative z-10">
            <div className={`text-[10px] font-bold uppercase ${colorStyles.label} mb-0.5`}>
              下翼 (买入 {lowerLegType})
            </div>
            <div className={`text-lg font-bold ${colorStyles.text}`}>${butterfly.lower_strike}</div>
            <div className={`text-[10px] ${colorStyles.cost}`}>
              支出 ${butterfly.lower_cost.toFixed(2)}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
};

const ButterflyDashboard: React.FC = () => {
  // Default list simulating backend config
  const [tickers, setTickers] = useState<string[]>([
    'AAPL', 'TSLA', 'NVDA', 'AMD', 'AMZN',
    'GOOGL', 'MSFT', 'META', 'NFLX', 'AVGO',
    'ORCL', 'PLTR', 'MU', 'SPY', 'QQQ'

  ]);
  const [newTicker, setNewTicker] = useState('');
  const [dataMap, setDataMap] = useState<Record<string, any>>({});
  const [loadingMap, setLoadingMap] = useState<Record<string, boolean>>({});
  const [isAutoRefreshing, setIsAutoRefreshing] = useState(false);

  // Function to fetch data for a single ticker
  const fetchTickerData = async (ticker: string) => {
    if (loadingMap[ticker]) return;

    setLoadingMap(prev => ({ ...prev, [ticker]: true }));
    try {
      const response = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker })
      });
      const json = await response.json();
      if (json.success) {
        setDataMap(prev => ({ ...prev, [ticker]: json.data }));
      } else {
        console.error(`Error fetching ${ticker}:`, json.error);
      }
    } catch (error) {
      console.error(`Failed to fetch ${ticker}:`, error);
    } finally {
      setLoadingMap(prev => ({ ...prev, [ticker]: false }));
    }
  };

  // Fetch all one by one to avoid backend congestion
  const fetchAll = async () => {
    setIsAutoRefreshing(true);
    // Execute sequentially
    for (const t of tickers) {
      await fetchTickerData(t);
    }
    setIsAutoRefreshing(false);
  };

  // Initial fetch
  useEffect(() => {
    fetchAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const addTicker = () => {
    if (newTicker && !tickers.includes(newTicker.toUpperCase())) {
      const t = newTicker.toUpperCase();
      setTickers([...tickers, t]);
      setNewTicker('');
      fetchTickerData(t);
    }
  };

  const removeTicker = (t: string) => {
    setTickers(tickers.filter(ticker => ticker !== t));
    const newData = { ...dataMap };
    delete newData[t];
    setDataMap(newData);
  };

  return (
    <div className="w-full max-w-[1600px] mx-auto p-6">
      {/* Header / Controls */}
      <div className="flex flex-col md:flex-row justify-between items-center mb-8 bg-white p-4 rounded-xl shadow-sm border border-gray-100">
        <div className="flex items-center gap-3 mb-4 md:mb-0">
          <Activity className="w-6 h-6 text-indigo-600" />
          <h1 className="text-2xl font-bold text-gray-800">
            市场热力图监控 <span className="text-sm font-normal text-gray-400 ml-2">Butterfly Heatmap v2.1</span>
          </h1>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center bg-gray-100 rounded-lg p-1">
            <input
              type="text"
              className="bg-transparent border-none focus:ring-0 text-sm px-3 py-1 w-24 outline-none"
              placeholder="代码"
              value={newTicker}
              onChange={e => setNewTicker(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && addTicker()}
            />
            <button onClick={addTicker} className="p-1 hover:bg-gray-200 rounded text-indigo-600">
              <Plus className="w-4 h-4" />
            </button>
          </div>

          <button
            onClick={fetchAll}
            disabled={isAutoRefreshing}
            className={`flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors ${isAutoRefreshing ? 'opacity-70 cursor-not-allowed' : ''}`}
          >
            <RefreshCw className={`w-4 h-4 ${isAutoRefreshing ? 'animate-spin' : ''}`} />
            {isAutoRefreshing ? '扫描中...' : '刷新全部分析'}
          </button>
        </div>
      </div>

      {/* Grid Layout */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {tickers.map(ticker => (
          <div key={ticker} className="relative group">
            <StrategyCard
              data={dataMap[ticker]}
              loading={loadingMap[ticker] || (!dataMap[ticker] && !loadingMap[ticker] && isAutoRefreshing)}
            />
            {/* Remove button (visible on hover) */}
            <button
              onClick={(e) => { e.stopPropagation(); removeTicker(ticker); }}
              className="absolute -top-2 -right-2 bg-gray-200 hover:bg-red-500 hover:text-white text-gray-500 rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity shadow-sm z-10"
              title="Remove ticker"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
        ))}

        {/* Add New Placeholder */}
        <div
          onClick={() => document.querySelector('input')?.focus()}
          className="border-2 border-dashed border-gray-300 rounded-lg p-6 flex flex-col items-center justify-center text-gray-400 hover:border-indigo-400 hover:text-indigo-500 cursor-pointer transition-colors min-h-[250px]"
        >
          <Plus className="w-8 h-8 mb-2" />
          <span className="text-sm font-medium">添加自选股</span>
        </div>
      </div>

      {tickers.length === 0 && (
        <div className="text-center py-20 text-gray-400">
          <AlertCircle className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>监控列表为空，请添加股票代码</p>
        </div>
      )}
    </div>
  );
};

export default ButterflyDashboard;