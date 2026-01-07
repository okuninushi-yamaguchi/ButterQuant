import React, { useState, useEffect } from 'react';
import { ArrowRight, Trophy, TrendingUp, AlertTriangle, Loader, ChevronDown } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { config } from '../config';

interface RankItem {
    rank: number;
    ticker: string;
    score: number;
    strategy: string;
    recommendation: string;
    confidence: string;
    date: string;
    tags: string[];
}

interface StrategyRankProps {
    onAnalyze: (ticker: string) => void;
}

const StrategyRank: React.FC<StrategyRankProps> = ({ onAnalyze }) => {
    const { t } = useTranslation();
    const [rankings, setRankings] = useState<RankItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [activeTab, setActiveTab] = useState<'ALL' | 'NASDAQ' | 'SP500'>('ALL');
    const [displayCount, setDisplayCount] = useState(20);

    useEffect(() => {
        fetchRankings();
    }, []);

    const fetchRankings = async () => {
        try {
            setLoading(true);
            // 获取所有数据以便客户端过滤
            const res = await fetch(`${config.API_URL}/api/rankings?limit=1000`);
            const json = await res.json();
            if (json.success) {
                setRankings(json.data);
            }
        } catch (error) {
            console.error('Failed to fetch rankings', error);
        } finally {
            setLoading(false);
        }
    };

    const filteredRankings = rankings.filter(item => {
        if (activeTab === 'ALL') return true;
        return item.tags?.includes(activeTab);
    });

    const getScoreColor = (score: number) => {
        if (score >= 80) return 'text-green-600 bg-green-50 border-green-200';
        if (score >= 60) return 'text-blue-600 bg-blue-50 border-blue-200';
        if (score >= 40) return 'text-amber-600 bg-amber-50 border-amber-200';
        return 'text-red-600 bg-red-50 border-red-200';
    };

    const getRecColor = (rec: string) => {
        switch (rec) {
            case 'STRONG_BUY': return 'text-green-700 bg-green-100';
            case 'BUY': return 'text-blue-700 bg-blue-100';
            case 'NEUTRAL': return 'text-gray-700 bg-gray-100';
            case 'AVOID': return 'text-red-700 bg-red-100';
            default: return 'text-gray-700 bg-gray-100';
        }
    };

    return (
        <div className="w-full max-w-7xl mx-auto p-6">
            <div className="flex flex-col md:flex-row justify-between items-center mb-8 bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <div className="flex items-center gap-4 mb-4 md:mb-0">
                    <div className="p-3 bg-amber-100 rounded-lg text-amber-600">
                        <Trophy className="w-8 h-8" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold text-gray-800">{t('rank.title')}</h1>
                        <p className="text-gray-500 text-sm">{t('rank.subtitle')}</p>
                    </div>
                </div>

                <div className="flex bg-gray-100 p-1 rounded-lg">
                    {(['ALL', 'NASDAQ', 'SP500'] as const).map((tab) => (
                        <button
                            key={tab}
                            onClick={() => { setActiveTab(tab); setDisplayCount(20); }}
                            className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${activeTab === tab
                                ? 'bg-white text-indigo-600 shadow-sm'
                                : 'text-gray-500 hover:text-gray-700'
                                }`}
                        >
                            {tab === 'ALL' ? t('rank.all_markets') : tab}
                        </button>
                    ))}
                </div>
            </div>

            <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                {loading ? (
                    <div className="flex flex-col items-center justify-center p-20 text-gray-400">
                        <Loader className="w-10 h-10 animate-spin mb-4 text-indigo-500" />
                        <p>{t('common.loading')}</p>
                    </div>
                ) : filteredRankings.length === 0 ? (
                    <div className="flex flex-col items-center justify-center p-20 text-gray-400">
                        <AlertTriangle className="w-12 h-12 mb-4 opacity-30" />
                        <p>{t('rank.no_data')}</p>
                    </div>
                ) : (
                    <div>
                        <div className="overflow-x-auto">
                            <table className="w-full">
                                <thead className="bg-gray-50 border-b border-gray-100">
                                    <tr>
                                        <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider w-20">
                                            {t('rank.rank')}
                                        </th>
                                        <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">
                                            {t('rank.ticker')}
                                        </th>
                                        <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">
                                            {t('rank.score')}
                                        </th>
                                        <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">
                                            {t('rank.strategy')}
                                        </th>
                                        <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">
                                            {t('rank.recommendation')}
                                        </th>
                                        <th className="px-6 py-4 text-right text-xs font-semibold text-gray-500 uppercase tracking-wider">
                                            {t('rank.action')}
                                        </th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-gray-100">
                                    {filteredRankings.slice(0, displayCount).map((item, idx) => (
                                        <tr
                                            key={item.ticker}
                                            className="hover:bg-gray-50 transition-colors group cursor-pointer"
                                            onClick={() => onAnalyze(item.ticker)}
                                        >
                                            <td className="px-6 py-4 whitespace-nowrap">
                                                <div className={`
                          w-8 h-8 flex items-center justify-center rounded-full font-bold text-sm
                          ${idx < 3 ? 'bg-indigo-600 text-white' : 'bg-gray-100 text-gray-500'}
                        `}>
                                                    {idx + 1}
                                                </div>
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap">
                                                <div className="flex items-center">
                                                    <span className="font-bold text-gray-900 text-lg">{item.ticker}</span>
                                                    {item.tags?.slice(0, 1).map(tag => (
                                                        <span key={tag} className="ml-2 text-[10px] bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded border border-gray-200">
                                                            {tag}
                                                        </span>
                                                    ))}
                                                </div>
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap">
                                                <div className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${getScoreColor(item.score)}`}>
                                                    {item.score.toFixed(1)}
                                                </div>
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap">
                                                <div className="flex items-center">
                                                    <span className={`text-sm font-medium ${item.strategy === 'CALL' ? 'text-green-600' :
                                                        item.strategy === 'PUT' ? 'text-red-600' : 'text-blue-600'
                                                        }`}>
                                                        {item.strategy}
                                                    </span>
                                                </div>
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap">
                                                <span className={`px-2 py-1 text-xs font-bold rounded ${getRecColor(item.recommendation)}`}>
                                                    {item.recommendation === 'STRONG_BUY' ? t('rank.rec_strong_buy') :
                                                        item.recommendation === 'BUY' ? t('rank.rec_buy') :
                                                            item.recommendation === 'NEUTRAL' ? t('rank.rec_neutral') :
                                                                item.recommendation === 'AVOID' ? t('rank.rec_avoid') :
                                                                    item.recommendation}
                                                </span>
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                                                <button
                                                    className="text-indigo-600 hover:text-indigo-900 group-hover:translate-x-1 transition-transform inline-flex items-center"
                                                    onClick={(e) => { e.stopPropagation(); onAnalyze(item.ticker); }}
                                                >
                                                    {t('rank.analyze')} <ArrowRight className="w-4 h-4 ml-1" />
                                                </button>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>

                        {filteredRankings.length > displayCount && (
                            <div className="p-4 border-t border-gray-100 bg-gray-50 flex justify-center">
                                <button
                                    onClick={() => setDisplayCount(prev => prev + 20)}
                                    className="flex items-center gap-2 px-6 py-2 bg-white border border-gray-300 rounded-lg text-gray-600 hover:bg-gray-50 hover:text-gray-900 transition-colors shadow-sm font-medium text-sm"
                                >
                                    {t('rank.load_more')} <ChevronDown className="w-4 h-4" />
                                </button>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default StrategyRank;
