# -*- coding: utf-8 -*-
"""
Daily Scanner - 每日批量扫描脚本
独立运行，不依赖Flask Context，但使用相同的Analyzer逻辑
"""

import os
import sys
import json
import yaml
import time
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加backend路径到sys.path以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.analyzer import ButterflyAnalyzer
from backend.database import DatabaseManager
from backend.deep_analysis_db import DeepAnalysisDB
from backend.ticker_utils import merge_ticker_lists, get_tickers_with_tags

# 加载配置
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

config = load_config()

# 配置日志
log_dir = config.get('paths', {}).get('log_dir', 'backend/logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    filename=os.path.join(log_dir, 'scanner.log'),
    level=getattr(logging, config.get('log', {}).get('level', 'INFO')),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

class DailyScanner:
    def __init__(self):
        self.config = config
        # 如果配置中有路径则使用，否则使用 DatabaseManager 默认
        db_path = config.get('paths', {}).get('db_path')
        self.db = DatabaseManager(db_path) if db_path else DatabaseManager()
        
        self.deep_db = DeepAnalysisDB(config.get('paths', {}).get('deep_db_path', 'backend/data/market_research.db'))
        self.data_dir = config.get('paths', {}).get('data_dir', 'backend/data')
        self.progress_file = os.path.join(self.data_dir, config.get('storage', {}).get('progress_file', 'scan_progress.txt'))
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def load_tickers_with_info(self):
        """加载所有待扫描的Ticker和标签"""
        # ... (unchanged)
        ticker_files = config.get('paths', {}).get('ticker_files', [])
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        file_map = {}
        for f in ticker_files:
            abs_path = os.path.join(base_dir, f)
            tag = 'UNKNOWN'
            if 'nas' in f.lower():
                tag = 'NASDAQ'
            elif 'sp500' in f.lower():
                tag = 'SP500'
            file_map[tag] = abs_path
            
        ticker_tags = get_tickers_with_tags(file_map)
        logging.info(f"Loaded {len(ticker_tags)} unique tickers with tags.")
        return ticker_tags

    def analyze_ticker(self, ticker, tags):
        """分析单个Ticker"""
        try:
            logging.info(f"Analyzing {ticker}...")
            analyzer = ButterflyAnalyzer(ticker)
            result = analyzer.full_analysis()
            
            # 简单的验证
            if result and 'score' in result:
                result['tags'] = tags  # 添加标签
                return result
            else:
                logging.warning(f"Analysis for {ticker} returned empty result.")
                return None
                
        except Exception as e:
            logging.error(f"Error analyzing {ticker}: {str(e)}")
            return None

    def save_results_to_json(self, results):
        """保存结果到JSON文件"""
        if not results:
            return
            
        # 按分数排序
        sorted_results = sorted(results, key=lambda x: x['score']['total'], reverse=True)
        
        # 简化版结果用于列表显示
        simplified_results = []
        for r in sorted_results:
            simplified_results.append({
                'rank': 0, # 这里暂时占位，后面生成
                'ticker': r['ticker'],
                'score': r['score']['total'],
                'strategy': r['fourier']['butterfly_type'],
                'recommendation': r['trade_suggestion']['action'],
                'confidence': r['score']['confidence_level'],
                'date': r['analysis_date'],
                'tags': r.get('tags', [])
            })
            
        # 添加排名
        for i, r in enumerate(simplified_results):
            r['rank'] = i + 1
            
        # 保存完整榜单
        full_path = os.path.join(self.data_dir, config.get('storage', {}).get('ranking_file', 'rankings_combined.json'))
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, indent=2, ensure_ascii=False)
            
        # 保存Top 20
        top20_path = os.path.join(self.data_dir, config.get('storage', {}).get('top20_file', 'rankings_top20.json'))
        with open(top20_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_results[:20], f, indent=2, ensure_ascii=False)
            
        logging.info(f"Saved rankings to {full_path} and {top20_path}")

    def run(self):
        """运行完整扫描流程"""
        start_time = datetime.now()
        logging.info("Starting Daily Scan...")
        
        ticker_tags = self.load_tickers_with_info()
        tickers = list(ticker_tags.keys())
        
        analyzed_count = 0
        success_count = 0
        results = []
        
        max_workers = config.get('scanner', {}).get('max_workers', 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(self.analyze_ticker, t, ticker_tags[t]): t for t in tickers}
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                analyzed_count += 1
                try:
                    data = future.result()
                    if data:
                        results.append(data)
                        success_count += 1
                        
                        # 保存到常规历史数据库 (JSON Blob)
                        if config.get('storage', {}).get('save_db', True):
                            self.db.save_analysis(data)

                        # 保存到深度分析数据库 (Flattened Rows)
                        # 注意：每个save_metric调用都会创建新连接，所以是线程安全的
                        self.deep_db.save_metric(data)
                            
                except Exception as e:
                    logging.error(f"Worker exception for {ticker}: {e}")
                    
                # 打印进度
                if analyzed_count % 5 == 0:
                    logging.info(f"Progress: {analyzed_count}/{len(tickers)} ({success_count} success)")

        # 保存JSON缓存
        if config.get('storage', {}).get('save_json', True):
            self.save_results_to_json(results)
            
        duration = datetime.now() - start_time
        logging.info(f"Scan completed in {duration}. Total: {len(tickers)}, Success: {success_count}")

if __name__ == "__main__":
    scanner = DailyScanner()
    scanner.run()
