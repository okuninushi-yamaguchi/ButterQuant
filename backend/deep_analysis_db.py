import sqlite3
import json
import os
from datetime import datetime

class DeepAnalysisDB:
    def __init__(self, db_path='backend/data/market_research.db'):
        self.db_path = db_path
        self._ensure_data_dir()
        self.init_db()

    def _ensure_data_dir(self):
        directory = os.path.dirname(self.db_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create a wide table for flattened metrics
        # This schema maps closely to the detailed JSON structure in talk.md
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            analysis_date TEXT NOT NULL,
            
            -- Basic Price Info
            current_price REAL,
            forecast_price REAL,
            price_stability REAL,
            
            -- Fourier Analysis
            trend_direction TEXT,
            trend_slope REAL,
            cycle_position TEXT,
            dominant_period_days REAL,
            period_strength REAL,
            
            -- GARCH & Volatility
            predicted_vol REAL,
            current_iv REAL,
            iv_percentile REAL,
            vol_mispricing REAL,
            iv_skew_call REAL,
            iv_skew_put REAL,
            
            -- Butterfly Strategy Mechanics
            strategy_type TEXT,
            center_strike REAL,
            wing_width REAL,
            profit_ratio REAL,
            max_profit REAL,
            max_loss REAL,
            prob_profit REAL,
            
            -- Greeks
            delta REAL,
            gamma REAL,
            vega REAL,
            theta REAL,
            
            -- Scores & Recommendation
            total_score REAL,
            score_price_match REAL,
            score_vol_mispricing REAL,
            score_stability REAL,
            score_fourier_align REAL,
            
            recommendation TEXT,
            confidence_level TEXT,
            
            -- Meta
            tags TEXT, 
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Indexes for common queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ticker ON daily_metrics (ticker)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON daily_metrics (analysis_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_score ON daily_metrics (total_score)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategy ON daily_metrics (strategy_type)')
        
        conn.commit()
        conn.close()

    def save_metric(self, result):
        """
        Flatten the complex analysis result dict into a single row
        """
        if not result:
            return

        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Flattening logic
            # Use .get() with defaults to avoid KeyErrors on missing data
            
            # 1. Basic
            ticker = result.get('ticker')
            analysis_date = result.get('analysis_date')
            current_price = result.get('current_price')
            price_stab = result.get('price_stability')
            
            # 2. Fourier
            fourier = result.get('fourier', {})
            trend_dir = fourier.get('trend_direction')
            trend_slope = fourier.get('trend_slope')
            cycle_pos = fourier.get('cycle_position')
            dom_period = fourier.get('dominant_period_days')
            period_str = fourier.get('period_strength')

            # 3. ARIMA (Skip detailed arrays, just use forecast price which is top level or in arima)
            # forecast_price is at top level
            forecast_price = result.get('forecast_price')
            
            # 4. GARCH
            garch = result.get('garch', {})
            pred_vol = garch.get('predicted_vol')
            curr_iv = garch.get('current_iv')
            iv_pct = garch.get('iv_percentile')
            vol_bais = garch.get('vol_mispricing')
            skew = garch.get('iv_skew', {})
            skew_c = skew.get('skew_call')
            skew_p = skew.get('skew_put')
            
            # 5. Butterfly
            bf = result.get('butterfly', {})
            strat_type = fourier.get('butterfly_type') # Type comes from Fourier logic usually
            center_k = bf.get('center_strike')
            width = bf.get('wing_width')
            prof_ratio = bf.get('profit_ratio')
            max_prof = bf.get('max_profit')
            max_loss = bf.get('max_loss')
            prob_win = bf.get('prob_profit')
            
            greeks = bf.get('greeks', {})
            delta = greeks.get('delta')
            gamma = greeks.get('gamma')
            vega = greeks.get('vega')
            theta = greeks.get('theta')
            
            # 6. Scores
            score_data = result.get('score', {})
            # If score is just a number (legacy), handle it, but per talk.md it's a dict
            if isinstance(score_data, (int, float)):
                total_score = score_data
                comps = {}
                confidence = 'UNKNOWN'
                rec = 'UNKNOWN'
            else:
                total_score = score_data.get('total')
                comps = score_data.get('components', {})
                confidence = score_data.get('confidence_level')
                # Recommendation is in trade_suggestion or score
                rec = score_data.get('recommendation')
            
            s_price = comps.get('price_match')
            s_vol = comps.get('vol_mispricing')
            s_stab = comps.get('stability')
            s_four = comps.get('fourier_align')
            
            # 7. Meta
            # Tags might be passed in separately or added to result
            tags_list = result.get('tags', [])
            tags_str = ','.join(tags_list) if tags_list else ''

            sql = '''
            INSERT INTO daily_metrics (
                ticker, analysis_date, current_price, forecast_price, price_stability,
                trend_direction, trend_slope, cycle_position, dominant_period_days, period_strength,
                predicted_vol, current_iv, iv_percentile, vol_mispricing, iv_skew_call, iv_skew_put,
                strategy_type, center_strike, wing_width, profit_ratio, max_profit, max_loss, prob_profit,
                delta, gamma, vega, theta,
                total_score, score_price_match, score_vol_mispricing, score_stability, score_fourier_align,
                recommendation, confidence_level, tags
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?
            )
            '''
            
            values = (
                ticker, analysis_date, current_price, forecast_price, price_stab,
                trend_dir, trend_slope, cycle_pos, dom_period, period_str,
                pred_vol, curr_iv, iv_pct, vol_bais, skew_c, skew_p,
                strat_type, center_k, width, prof_ratio, max_prof, max_loss, prob_win,
                delta, gamma, vega, theta,
                total_score, s_price, s_vol, s_stab, s_four,
                rec, confidence, tags_str
            )
            
            cursor.execute(sql, values)
            conn.commit()
            
        except Exception as e:
            print(f"Failed to save to Deep Analysis DB for {result.get('ticker')}: {e}")
        finally:
            conn.close()
