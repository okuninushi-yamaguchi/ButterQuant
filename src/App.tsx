import React, { useState } from 'react';
import ButterflyOptionAnalyzer from './components/OptionAnalyzer';
import ButterflyDashboard from './components/Dashboard';
import { LayoutGrid, Microscope, Languages } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import logo from './assets/logo_remove_background.png';

export default function App() {
  const [view, setView] = useState<'dashboard' | 'analyzer'>('dashboard');
  const [selectedTicker, setSelectedTicker] = useState<string>('');
  const { t, i18n } = useTranslation();

  const toggleLanguage = () => {
    const newLang = i18n.language === 'zh' ? 'en' : 'zh';
    i18n.changeLanguage(newLang);
    localStorage.setItem('i18nextLng', newLang);
  };

  const handleTickerSelect = (ticker: string) => {
    setSelectedTicker(ticker);
    setView('analyzer');
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Navigation Bar */}
      <nav className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center gap-2">
                <img src={logo} alt="ButterQuant Logo" className="w-8 h-8 rounded-full object-cover" />
                <span className="font-bold text-xl text-gray-800 tracking-tight">ButterQuant</span>
              </div>
              <div className="hidden sm:ml-8 sm:flex sm:space-x-8">
                <button
                  onClick={() => setView('dashboard')}
                  className={`${view === 'dashboard'
                    ? 'border-indigo-500 text-gray-900'
                    : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                    } inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors duration-200`}
                >
                  <LayoutGrid className="w-4 h-4 mr-2" />
                  {t('nav.dashboard')}
                </button>
                <button
                  onClick={() => setView('analyzer')}
                  className={`${view === 'analyzer'
                    ? 'border-indigo-500 text-gray-900'
                    : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                    } inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors duration-200`}
                >
                  <Microscope className="w-4 h-4 mr-2" />
                  {t('nav.analyzer')}
                </button>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <button
                onClick={toggleLanguage}
                className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-gray-700 bg-gray-50 border border-gray-300 rounded-md hover:bg-gray-100 transition-colors"
              >
                <Languages className="w-4 h-4" />
                {i18n.language === 'zh' ? 'English' : '中文'}
              </button>
              <span className="hidden md:inline text-xs text-gray-400 border border-gray-200 px-2 py-1 rounded">
                {t('nav.status')}
              </span>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <div className={view === 'dashboard' ? 'block' : 'hidden'}>
          <ButterflyDashboard onAnalyzeTicker={handleTickerSelect} />
        </div>

        {view === 'analyzer' && (
          <ButterflyOptionAnalyzer initialTicker={selectedTicker} />
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 py-6 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row justify-between items-center text-sm text-gray-500">
          <div className="flex items-center gap-2 mb-2 md:mb-0">
            <img src={logo} alt="ButterQuant" className="w-5 h-5 rounded-full object-cover" />
            <span className="font-semibold text-gray-700">ButterQuant</span>
            <span>© 2025</span>
          </div>
          <div className="flex gap-6">
            <a href="#" className="hover:text-indigo-600 transition-colors">Documentation</a>
            <a href="#" className="hover:text-indigo-600 transition-colors">Privacy</a>
            <a href="#" className="hover:text-indigo-600 transition-colors">Terms</a>
          </div>
        </div>
      </footer>
    </div>
  );
}