import os

def load_tickers_from_markdown(file_path):
    """
    从Markdown文件加载股票代码
    假设格式为每行一个代码，或者列表形式
    """
    tickers = []
    if not os.path.exists(file_path):
        print(f"Warning: Ticker file not found at {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 忽略Markdown标题和列表标记
        if line.startswith('#'):
            continue
        if line.startswith('- '):
            line = line[2:]
        if line.startswith('* '):
            line = line[2:]
            
        # 提取代码 (假设代码是行中第一个单词，或者是纯代码)
        # 有些文件可能有 "AAPL - Apple Inc." 格式
        parts = line.split(' ')
        if parts:
            ticker = parts[0].strip().upper()
            # 简单的验证：纯字母且长度在1-5之间 (美股)
            if ticker.isalpha() and 1 <= len(ticker) <= 5:
                tickers.append(ticker)
                
    # 去重
    return list(set(tickers))

def get_tickers_with_tags(file_map):
    """
    加载Ticker并附带标签
    file_map: {'NAS100': 'path/to/nas100.md', 'SP500': 'path/to/sp500.md'}
    Returns: {'AAPL': ['NAS100', 'SP500'], ...}
    """
    ticker_tags = {}
    
    for tag, path in file_map.items():
        tickers = load_tickers_from_markdown(path)
        for t in tickers:
            if t not in ticker_tags:
                ticker_tags[t] = []
            if tag not in ticker_tags[t]:
                ticker_tags[t].append(tag)
                
    return ticker_tags

def merge_ticker_lists(files):
    """合并多个来源的Ticker"""
    all_tickers = set()
    for f in files:
        tickers = load_tickers_from_markdown(f)
        all_tickers.update(tickers)
    return sorted(list(all_tickers))

if __name__ == "__main__":
    # Test
    t = load_tickers_from_markdown('../nas100.md')
    print(f"Loaded {len(t)} tickers from nas100.md")
