# -*- coding: utf-8 -*-
"""
ARIMA-GARCH 蝴蝶期权分析后端 API (改进版)
依赖安装: pip install flask flask-cors yfinance numpy pandas scipy statsmodels arch
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import warnings
from analyzer import ButterflyAnalyzer
from database import DatabaseManager
import os
import json

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global scanner instance status
SCAN_STATUS = {
    'is_scanning': False,
    'last_scan': None,
    'progress': 'Idle'
}

def run_scanner_background():
    """后台运行扫描"""
    global SCAN_STATUS
    try:
        from daily_scanner import DailyScanner
        SCAN_STATUS['is_scanning'] = True
        SCAN_STATUS['progress'] = 'Starting...'
        
        scanner = DailyScanner()
        scanner.run()
        
        SCAN_STATUS['last_scan'] = datetime.now().isoformat()
        SCAN_STATUS['progress'] = 'Completed'
    except Exception as e:
        SCAN_STATUS['progress'] = f'Error: {str(e)}'
    finally:
        SCAN_STATUS['is_scanning'] = False

@app.route('/api/rankings', methods=['GET'])
def get_rankings():
    """获取排行榜数据 (优先读取数据库，实现实时更新)"""
    try:
        limit = request.args.get('limit', default=100, type=int)
        
        # 尝试从数据库获取实时数据
        try:
            # 使用 DatabaseManager 默认路径（已修改为绝对路径）
            db = DatabaseManager()
            data = db.get_latest_ranking(limit=limit)
            if data and len(data) > 0:
                # 补充排名
                for i, item in enumerate(data):
                    item['rank'] = i + 1
                return jsonify({'success': True, 'data': data})
        except Exception as db_err:
            print(f"Database read error: {db_err}")
            # Fallback to JSON file if DB fails
            pass

        data_file = 'backend/data/rankings_combined.json'
        
        if not os.path.exists(data_file):
            return jsonify({'success': False, 'error': 'No ranking data available'}), 404
            
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if limit:
            data = data[:limit]
            
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/rankings/top20', methods=['GET'])
def get_top20_rankings():
    """获取Top 20排行榜 (优先读取数据库)"""
    try:
        # 尝试从数据库获取
        try:
            # 使用 DatabaseManager 默认路径（已修改为绝对路径）
            db = DatabaseManager()
            data = db.get_latest_ranking(limit=20)
            if data and len(data) > 0:
                for i, item in enumerate(data):
                    item['rank'] = i + 1
                return jsonify({'success': True, 'data': data})
        except Exception:
            pass

        data_file = 'backend/data/rankings_top20.json'
        
        if not os.path.exists(data_file):
            # 尝试读取完整榜单并截取
            full_file = 'backend/data/rankings_combined.json'
            if os.path.exists(full_file):
                with open(full_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return jsonify({'success': True, 'data': data[:20]})
            return jsonify({'success': False, 'error': 'No ranking data available'}), 404
            
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/scan', methods=['POST'])
def trigger_scan():
    """触发后台扫描"""
    global SCAN_STATUS
    if SCAN_STATUS['is_scanning']:
        return jsonify({'success': False, 'message': 'Scan already in progress'}), 409
        
    import threading
    thread = threading.Thread(target=run_scanner_background)
    thread.start()
    
    return jsonify({'success': True, 'message': 'Scan started in background'})

@app.route('/api/scan/status', methods=['GET'])
def get_scan_status():
    """获取扫描状态"""
    return jsonify(SCAN_STATUS)


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