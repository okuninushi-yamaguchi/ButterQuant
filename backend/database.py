import sqlite3
import json
import os
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path=None):
        if db_path is None:
            # 默认路径相对于当前文件 (database.py) 的位置
            # database.py 在 backend/ 下，所以 data/ 在同一目录下
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.db_path = os.path.join(base_dir, 'data', 'history.db')
        else:
            self.db_path = db_path
            
        self._ensure_data_dir()
        self.init_db()

    def _ensure_data_dir(self):
        """确保数据目录存在"""
        directory = os.path.dirname(self.db_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    def get_connection(self):
        """获取数据库连接"""
        return sqlite3.connect(self.db_path)

    def init_db(self):
        """初始化数据库表"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 分析历史表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            analysis_date TEXT NOT NULL,
            total_score REAL,
            butterfly_type TEXT,
            recommendation TEXT,
            full_result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 索引（加速查询）
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ticker ON analysis_history (ticker)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON analysis_history (analysis_date)')
        
        conn.commit()
        conn.close()

    def save_analysis(self, result):
        """保存分析结果"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # 提取关键字段
            ticker = result.get('ticker')
            analysis_date = result.get('analysis_date')
            score = result.get('score', {}).get('total', 0)
            butterfly_type = result.get('fourier', {}).get('butterfly_type', 'UNKNOWN')
            recommendation = result.get('trade_suggestion', {}).get('action', 'NEUTRAL')
            
            # 序列化完整结果
            full_result_json = json.dumps(result, ensure_ascii=False)
            
            cursor.execute('''
            INSERT INTO analysis_history (
                ticker, analysis_date, total_score, butterfly_type, recommendation, full_result
            ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (ticker, analysis_date, score, butterfly_type, recommendation, full_result_json))
            
            conn.commit()
            print(f"[{ticker}] 分析结果已保存到数据库")
            
        except Exception as e:
            print(f"保存数据库失败: {e}")
        finally:
            conn.close()

    def get_history(self, ticker, limit=5):
        """获取某股票的历史分析"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT analysis_date, total_score, butterfly_type, recommendation, full_result
        FROM analysis_history
        WHERE ticker = ?
        ORDER BY analysis_date DESC
        LIMIT ?
        ''', (ticker, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                'analysis_date': row[0],
                'total_score': row[1],
                'butterfly_type': row[2],
                'recommendation': row[3],
                'full_result': json.loads(row[4])
            })
            
        return history
    
    def get_latest_ranking(self, limit=20):
        """获取最近一次分析的排名（用于API）"""
        # 注意：这只是简单的从历史中取最近的，实际排行榜可能需要单独的表或更复杂的查询
        # 这里仅作为数据库查询示例，实际应用可能会用 data/rankings_xx.json
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 获取每个ticker最近的一次分析，并包含 full_result 以获取 tags
        # 使用子查询确保取到的是每只股票最新日期的那一行数据
        query = '''
        SELECT t1.ticker, t1.analysis_date, t1.total_score, t1.butterfly_type, t1.recommendation, t1.full_result
        FROM analysis_history t1
        INNER JOIN (
            SELECT ticker, MAX(analysis_date) as max_date
            FROM analysis_history
            GROUP BY ticker
        ) t2 ON t1.ticker = t2.ticker AND t1.analysis_date = t2.max_date
        ORDER BY t1.total_score DESC
        LIMIT ?
        '''
        
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
        conn.close()
        
        rankings = []
        for row in rows:
            try:
                full_data = json.loads(row[5]) if row[5] else {}
                tags = full_data.get('tags', [])
            except:
                tags = []

            rankings.append({
                'ticker': row[0],
                'analysis_date': row[1],
                'score': row[2],
                'strategy': row[3],  # 前端期望字段名为 strategy
                'recommendation': row[4],
                'tags': tags
            })
            
        return rankings
