import React, { useState } from 'react';
import ButterflyOptionAnalyzer from './components/ButterflyOptionAnalyzer';
import ButterflyDashboard from './components/ButterflyDashboard';
import { LayoutGrid, Microscope, Waves } from 'lucide-react';

export default function App() {
  const [view, setView] = useState<'dashboard' | 'analyzer'>('dashboard');

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Navigation Bar */}
      <nav className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center gap-2">
                <Waves className="w-8 h-8 text-indigo-600" />
                <span className="font-bold text-xl text-gray-800 tracking-tight">ButterQuant</span>
              </div>
              <div className="hidden sm:ml-8 sm:flex sm:space-x-8">
                <button
                  onClick={() => setView('dashboard')}
                  className={`${
                    view === 'dashboard'
                      ? 'border-indigo-500 text-gray-900'
                      : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                  } inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors duration-200`}
                >
                  <LayoutGrid className="w-4 h-4 mr-2" />
                  策略热力图
                </button>
                <button
                  onClick={() => setView('analyzer')}
                  className={`${
                    view === 'analyzer'
                      ? 'border-indigo-500 text-gray-900'
                      : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                  } inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors duration-200`}
                >
                  <Microscope className="w-4 h-4 mr-2" />
                  深度分析器
                </button>
              </div>
            </div>
            
            <div className="flex items-center">
              <span className="text-xs text-gray-400 border border-gray-200 px-2 py-1 rounded">
                Backend Status: Connected
              </span>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        {/* 
          We use CSS toggling (hidden/block) for the Dashboard to keep it mounted.
          This ensures the setInterval for background refreshing keeps running 
          even when the user is viewing the Analyzer.
        */}
        <div className={view === 'dashboard' ? 'block' : 'hidden'}>
          <ButterflyDashboard />
        </div>
        
        {/* We can conditionally render the Analyzer to save resources when not in use, 
            or keep it alive if we want to preserve chart state. 
            Here we unmount it to reset state on re-entry as per standard behavior. */}
        {view === 'analyzer' && (
          <ButterflyOptionAnalyzer />
        )}
      </main>
    </div>
  );
}