# 数据库维护指南 (Database Maintenance Guide)

ButterQuant 使用 SQLite 作为本地数据库，用于存储每日扫描的历史分析结果。

## 📁 数据库位置
- **路径**: `backend/data/history.db`
- **类型**: SQLite 3

## 📊 数据库结构 (Schema)

目前主要包含一张表 `analysis_history`：

| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `id` | INTEGER PRIMARY KEY | 自增主键 |
| `ticker` | TEXT | 股票代码 (如 AAPL) |
| `analysis_date` | TEXT | 分析时间 (ISO 8601 格式) |
| `total_score` | REAL | 策略总分 (0-100) |
| `butterfly_type` | TEXT | 策略类型 (CALL/PUT/IRON) |
| `recommendation` | TEXT | 建议 (STRONG_BUY/BUY/NEUTRAL/AVOID) |
| `full_result` | TEXT (JSON) | 完整的分析结果 JSON 字符串 |
| `created_at` | TIMESTAMP | 记录创建时间 |

## 📁 深度分析数据库 (Deep Analysis DB)
- **路径**: `backend/data/market_research.db`
- **用途**: 存储展开的（Flattened）每日分析数据，便于 SQL 查询和深度挖掘。
- **表名**: `daily_metrics`
- **主要字段**:
    - `ticker`, `analysis_date`, `current_price`
    - `trend_direction`, `trend_slope` (傅立叶趋势)
    - `predicted_vol`, `vol_mispricing` (波动率)
    - `delta`, `gamma`, `vega` (Greeks)
    - `profit_ratio`, `prob_profit` (策略指标)

此数据库设计为宽表模式，适合直接导入 Pandas 进行批量分析：
```python
import sqlite3
import pandas as pd
conn = sqlite3.connect('backend/data/market_research.db')
df = pd.read_sql("SELECT * FROM daily_metrics WHERE total_score > 80", conn)
```

## 🛠️ 常用维护操作

### 1. 备份数据库
建议定期备份数据库文件，以防数据丢失。
```bash
# Windows
copy backend\data\history.db backend\data\history_backup.db
```

### 2. 查看数据统计
使用 python 脚本或 SQLite 客户端工具。

**Python 示例:**
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('backend/data/history.db')

# 查看总记录数
count = pd.read_sql_query("SELECT COUNT(*) FROM analysis_history", conn)
print(f"Total records: {count.iloc[0,0]}")

# 查看最近的高分机会
top_picks = pd.read_sql_query("""
    SELECT ticker, total_score, butterfly_type, analysis_date 
    FROM analysis_history 
    WHERE total_score > 80 
    ORDER BY analysis_date DESC LIMIT 10
""", conn)
print(top_picks)

conn.close()
```

### 3. 清理旧数据
如果数据库文件过大，可以删除过期的历史记录（例如保留最近60天）。

**SQL 示例:**
```sql
DELETE FROM analysis_history 
WHERE analysis_date < date('now', '-60 days');

VACUUM; -- 释放未使用的空间
```

### 4. 检查数据库完整性
如果遇到数据库错误，可以运行检查。
```sql
PRAGMA integrity_check;
```

## 🔄 迁移与升级
如果将来需要添加新字段或迁移到 PostgreSQL/MySQL：
1. 使用 `backend/database.py` 中的 `init_db` 函数修改表结构。
2. 编写迁移脚本将数据从 SQLite 导出并导入新数据库。

## 📝 开发说明
数据库交互逻辑封装在 `backend/database.py` 的 `DatabaseManager` 类中。
- `save_analysis(result)`: 保存单条分析结果。
- `get_history(ticker)`: 获取个股历史记录。
