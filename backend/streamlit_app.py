#!/usr/bin/env python3
import os
import sys

# 完全禁用 PyArrow
os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'
os.environ['STREAMLIT_SERVER_ENABLE_ARROW'] = 'false'

"""
Streamlit 数据可视化应用 - ButterQuant 数据库展示
纯 HTML 版本 - 完全不依赖 PyArrow
"""

import streamlit as st
import pandas as pd
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from contextlib import contextmanager

# 设置页面配置
st.set_page_config(
    page_title="ButterQuant 数据分析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 数据库路径
DB_PATH = Path(__file__).parent / "data" / "history.db"

@contextmanager
def get_db_connection():
    """获取数据库连接（线程安全）"""
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()

@st.cache_data(ttl=300)
def load_data():
    """加载所有数据"""
    with get_db_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM analysis_history ORDER BY analysis_date DESC",
            conn
        )
    
    df['analysis_date'] = pd.to_datetime(df['analysis_date'])
    return df

def safe_json_display(data_dict, title=""):
    """安全显示JSON（使用HTML）"""
    if not data_dict:
        st.warning("无数据")
        return
    
    html = f"<h4>{title}</h4>" if title else ""
    html += "<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>"
    html += "<pre style='margin: 0; font-size: 12px;'>"
    html += json.dumps(data_dict, indent=2, ensure_ascii=False)
    html += "</pre></div>"
    st.markdown(html, unsafe_allow_html=True)

def render_dataframe_as_html(df, max_rows=100):
    """将DataFrame渲染为HTML表格"""
    display_df = df.head(max_rows)
    
    html = """
    <style>
        .dataframe-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .dataframe-table th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
            border: none;
        }
        .dataframe-table td {
            padding: 10px;
            border-bottom: 1px solid #e0e0e0;
        }
        .dataframe-table tr:hover {
            background-color: #f5f7fa;
        }
        .dataframe-table tr:nth-child(even) {
            background-color: #fafbfc;
        }
        .score-high { 
            color: #00CC44; 
            font-weight: bold;
            background-color: #e8f5e9;
            padding: 4px 8px;
            border-radius: 4px;
        }
        .score-medium { 
            color: #FF8800; 
            font-weight: bold;
            background-color: #fff3e0;
            padding: 4px 8px;
            border-radius: 4px;
        }
        .score-low { 
            color: #FF4444; 
            font-weight: bold;
            background-color: #ffebee;
            padding: 4px 8px;
            border-radius: 4px;
        }
        .rec-strong-buy { 
            background: linear-gradient(135deg, #00CC44 0%, #00AA33 100%);
            color: white; 
            padding: 6px 12px; 
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
        }
        .rec-buy { 
            background: linear-gradient(135deg, #00DD88 0%, #00BB66 100%);
            color: white; 
            padding: 6px 12px; 
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
        }
        .rec-neutral { 
            background: linear-gradient(135deg, #FFAA00 0%, #FF8800 100%);
            color: white; 
            padding: 6px 12px; 
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
        }
        .rec-avoid { 
            background: linear-gradient(135deg, #FF4444 0%, #CC0000 100%);
            color: white; 
            padding: 6px 12px; 
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
        }
    </style>
    <div style='overflow-x: auto;'>
    <table class="dataframe-table">
        <thead><tr>
    """
    
    for col in display_df.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"
    
    for _, row in display_df.iterrows():
        html += "<tr>"
        for col in display_df.columns:
            value = row[col]
            
            if col == 'total_score':
                score = float(value) if pd.notna(value) else 0
                if score >= 70:
                    css_class = 'score-high'
                elif score >= 50:
                    css_class = 'score-medium'
                else:
                    css_class = 'score-low'
                html += f'<td><span class="{css_class}">{score:.2f}</span></td>'
            elif col == 'recommendation':
                rec_map = {
                    'STRONG_BUY': 'rec-strong-buy',
                    'BUY': 'rec-buy',
                    'NEUTRAL': 'rec-neutral',
                    'AVOID': 'rec-avoid'
                }
                css_class = rec_map.get(value, '')
                display_value = value if pd.notna(value) else '-'
                html += f'<td><span class="{css_class}">{display_value}</span></td>'
            elif col == 'analysis_date':
                if pd.notna(value):
                    html += f'<td>{value.strftime("%Y-%m-%d %H:%M")}</td>'
                else:
                    html += '<td>-</td>'
            else:
                display_value = str(value) if pd.notna(value) else '-'
                html += f'<td>{display_value}</td>'
        html += "</tr>"
    
    html += "</tbody></table></div>"
    return html

# 标题
st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;'>
    <h1 style='color: white; margin: 0;'>📊 ButterQuant 数据分析平台</h1>
    <p style='color: rgba(255,255,255,0.9); margin: 10px 0 0 0;'>Pure HTML Version - Zero PyArrow Dependencies</p>
</div>
""", unsafe_allow_html=True)

# 加载数据
try:
    df = load_data()
    
    if df.empty:
        st.warning("❌ 数据库中没有数据")
        st.info("💡 请先运行分析器生成数据")
    else:
        # 侧边栏过滤器
        st.sidebar.markdown("## 🔍 筛选条件")
        
        tickers = sorted(df['ticker'].unique())
        selected_ticker = st.sidebar.multiselect(
            "选择股票代码",
            tickers,
            default=tickers[:5] if len(tickers) > 5 else tickers
        )
        
        score_range = st.sidebar.slider(
            "总分范围",
            float(df['total_score'].min()),
            float(df['total_score'].max()),
            (0.0, 100.0)
        )
        
        butterfly_types = sorted(df['butterfly_type'].dropna().unique())
        selected_types = st.sidebar.multiselect(
            "策略类型",
            butterfly_types,
            default=butterfly_types
        )
        
        recommendations = sorted(df['recommendation'].dropna().unique())
        selected_recommendations = st.sidebar.multiselect(
            "建议",
            recommendations,
            default=recommendations
        )
        
        date_range = st.sidebar.date_input(
            "分析日期范围",
            value=(df['analysis_date'].min().date(), df['analysis_date'].max().date()),
            min_value=df['analysis_date'].min().date(),
            max_value=df['analysis_date'].max().date()
        )
        
        # 应用过滤
        filtered_df = df[
            (df['ticker'].isin(selected_ticker)) &
            (df['total_score'].between(score_range[0], score_range[1])) &
            (df['butterfly_type'].isin(selected_types)) &
            (df['recommendation'].isin(selected_recommendations)) &
            (df['analysis_date'].dt.date >= date_range[0]) &
            (df['analysis_date'].dt.date <= date_range[1])
        ]
        
        # 统计信息
        st.sidebar.markdown("---")
        st.sidebar.markdown("## 📈 统计信息")
        st.sidebar.metric("记录总数", len(df))
        st.sidebar.metric("已筛选记录", len(filtered_df))
        if len(filtered_df) > 0:
            st.sidebar.metric("平均分数", f"{filtered_df['total_score'].mean():.2f}")
            st.sidebar.metric("最高分", f"{filtered_df['total_score'].max():.2f}")
        
        # 选项卡
        tab1, tab2, tab3, tab4 = st.tabs(["📊 概览", "📋 数据表", "📈 图表分析", "🔍 详细查询"])
        
        with tab1:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("总记录数", len(filtered_df))
            with col2:
                strong_buy = len(filtered_df[filtered_df['recommendation'] == 'STRONG_BUY'])
                st.metric("强买信号", strong_buy)
            with col3:
                buy = len(filtered_df[filtered_df['recommendation'] == 'BUY'])
                st.metric("买入信号", buy)
            with col4:
                avoid = len(filtered_df[filtered_df['recommendation'] == 'AVOID'])
                st.metric("回避信号", avoid)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if len(filtered_df) > 0:
                    recommendation_counts = filtered_df['recommendation'].value_counts()
                    fig_rec = px.pie(
                        values=recommendation_counts.values,
                        names=recommendation_counts.index,
                        title="建议分布",
                        color_discrete_map={
                            'STRONG_BUY': '#00CC44',
                            'BUY': '#00DD88',
                            'NEUTRAL': '#FFAA00',
                            'AVOID': '#FF4444'
                        }
                    )
                    st.plotly_chart(fig_rec, use_container_width=True)
            
            with col2:
                if len(filtered_df) > 0:
                    butterfly_counts = filtered_df['butterfly_type'].value_counts()
                    fig_butterfly = px.pie(
                        values=butterfly_counts.values,
                        names=butterfly_counts.index,
                        title="策略类型分布"
                    )
                    st.plotly_chart(fig_butterfly, use_container_width=True)
            
            st.markdown("---")
            
            if len(filtered_df) > 0:
                fig_hist = px.histogram(
                    filtered_df,
                    x='total_score',
                    nbins=20,
                    title="分数分布",
                    labels={'total_score': '总分', 'count': '数量'},
                    color_discrete_sequence=['#636EFA']
                )
                st.plotly_chart(fig_hist, use_container_width=True)
        
        with tab2:
            display_cols = ['ticker', 'analysis_date', 'total_score', 'butterfly_type', 'recommendation']
            
            st.markdown("### 数据表")
            
            col1, col2 = st.columns(2)
            with col1:
                sort_col = st.selectbox("排序列", display_cols)
            with col2:
                sort_order = st.radio("排序方式", ["降序", "升序"], horizontal=True)
            
            sorted_df = filtered_df.sort_values(
                sort_col,
                ascending=(sort_order == "升序")
            )
            
            display_df = sorted_df[display_cols].copy()
            html_table = render_dataframe_as_html(display_df, max_rows=100)
            st.markdown(html_table, unsafe_allow_html=True)
            
            st.markdown(f"**显示前 100 条记录，共 {len(display_df)} 条**")
            
            csv = display_df.head(1000).to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 下载为 CSV (前1000条)",
                data=csv,
                file_name=f"analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with tab3:
            st.markdown("### 📈 时间序列分析")
            
            if len(filtered_df) > 0:
                daily_avg = filtered_df.groupby(filtered_df['analysis_date'].dt.date)['total_score'].agg(['mean', 'max', 'min', 'count']).reset_index()
                daily_avg.columns = ['日期', '平均分', '最高分', '最低分', '记录数']
                
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=daily_avg['日期'],
                    y=daily_avg['平均分'],
                    mode='lines+markers',
                    name='平均分',
                    line=dict(color='#636EFA', width=2)
                ))
                fig_trend.add_trace(go.Scatter(
                    x=daily_avg['日期'],
                    y=daily_avg['最高分'],
                    mode='lines',
                    name='最高分',
                    line=dict(color='#00CC44', dash='dash')
                ))
                fig_trend.add_trace(go.Scatter(
                    x=daily_avg['日期'],
                    y=daily_avg['最低分'],
                    mode='lines',
                    name='最低分',
                    line=dict(color='#FF4444', dash='dash')
                ))
                fig_trend.update_layout(
                    title="分数时间序列",
                    xaxis_title="日期",
                    yaxis_title="分数",
                    hovermode='x unified',
                    height=400
                )
                st.plotly_chart(fig_trend, use_container_width=True)
                
                st.markdown("---")
                st.markdown("### 📊 股票代码分析")
                
                ticker_stats = filtered_df.groupby('ticker').agg({
                    'total_score': ['mean', 'count'],
                    'recommendation': lambda x: (x == 'STRONG_BUY').sum()
                }).round(2)
                ticker_stats.columns = ['平均分', '记录数', '强买次数']
                ticker_stats = ticker_stats.sort_values('平均分', ascending=False)
                
                fig_ticker = px.bar(
                    ticker_stats.reset_index(),
                    x='ticker',
                    y='平均分',
                    color='平均分',
                    title="各股票平均分",
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_ticker, use_container_width=True)
            else:
                st.warning("没有数据可供分析")
        
        with tab4:
            st.markdown("### 🔍 查询单条记录详情")
            
            if len(filtered_df) > 0:
                record_idx = st.selectbox(
                    "选择记录",
                    range(len(filtered_df)),
                    format_func=lambda i: f"{filtered_df.iloc[i]['ticker']} - {filtered_df.iloc[i]['analysis_date']} (分数: {filtered_df.iloc[i]['total_score']:.1f})"
                )
                
                selected_record = filtered_df.iloc[record_idx]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("股票代码", selected_record['ticker'])
                with col2:
                    st.metric("总分", f"{selected_record['total_score']:.2f}")
                with col3:
                    st.metric("策略类型", selected_record['butterfly_type'])
                with col4:
                    st.metric("建议", selected_record['recommendation'])
                
                st.markdown("---")
                st.markdown("### 📄 完整分析结果")
                
                if pd.notna(selected_record['full_result']):
                    try:
                        full_data = json.loads(selected_record['full_result'])
                        
                        detail_col1, detail_col2, detail_col3 = st.columns(3)
                        
                        with detail_col1:
                            if 'fourier' in full_data:
                                safe_json_display({
                                    "trend_direction": full_data['fourier'].get('trend_direction'),
                                    "trend_slope": full_data['fourier'].get('trend_slope'),
                                    "cycle_position": full_data['fourier'].get('cycle_position'),
                                }, "📊 傅立叶分析")
                        
                        with detail_col2:
                            if 'arima' in full_data:
                                safe_json_display({
                                    "forecast_7d": full_data['arima'].get('forecast_7d'),
                                    "forecast_30d": full_data['arima'].get('forecast_30d'),
                                }, "📈 ARIMA 预测")
                        
                        with detail_col3:
                            if 'garch' in full_data:
                                safe_json_display({
                                    "predicted_vol": full_data['garch'].get('predicted_vol'),
                                    "historical_vol": full_data['garch'].get('historical_vol'),
                                }, "📊 GARCH 波动率")
                        
                        st.markdown("---")
                        
                        if 'butterfly' in full_data:
                            st.subheader("🦋 期权蝴蝶策略详情")
                            butterfly = full_data['butterfly']
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("中心行权价", f"${butterfly.get('center_strike', 0):.2f}")
                            with col2:
                                st.metric("最大利润", f"${butterfly.get('max_profit', 0):.2f}")
                            with col3:
                                st.metric("最大亏损", f"${butterfly.get('max_loss', 0):.2f}")
                            with col4:
                                prob = butterfly.get('prob_profit', 0)
                                st.metric("获利概率", f"{prob*100:.1f}%" if prob else "N/A")
                        
                        with st.expander("查看完整 JSON 数据"):
                            safe_json_display(full_data, "完整数据")
                    
                    except Exception as e:
                        st.error(f"无法解析 JSON: {e}")
                else:
                    st.warning("此记录没有详细数据")
            else:
                st.warning("没有符合条件的记录")

except Exception as e:
    st.error(f"❌ 错误: {e}")
    import traceback
    with st.expander("查看详细错误信息"):
        st.code(traceback.format_exc())

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px; padding: 20px;'>
    <strong>ButterQuant 数据分析平台</strong><br>
    数据库路径: data/history.db | 最后更新: 2026-01-07<br>
    ✅ Pure HTML Version | ✅ Zero PyArrow Dependencies | ✅ Thread-Safe SQLite
</div>
""", unsafe_allow_html=True)