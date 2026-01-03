# ButterQuant 项目运行指南

本指南将协助您在本地环境成功运行 ButterQuant 项目（ARIMA-GARCH 蝴蝶期权分析器）。

## 1. 环境准备

在开始之前，请确保您的系统中已安装以下软件：
- **Python 3.9+**
- **Node.js (建议 v18+)**
- **npm** (随 Node.js 一起安装)

---

## 2. 后端设置 (Python Flask)

后端处理复杂的金融计算和数据获取。

### 步骤：
1. **进入后端目录**：
   ```powershell
   cd backend
   ```

2. **创建虚拟环境**（推荐）：
   ```powershell
   python -m venv venv
   ```

3. **安装依赖项**（建议使用具体路径）：
   ```powershell
   .\venv\Scripts\python -m pip install -r requirements.txt
   ```

4. **运行后端服务器**：
   ```powershell
   .\venv\Scripts\python app.py
   ```
   > [!NOTE]
   > 后端将运行在 `http://localhost:5000`。
   > 健康检查地址：`http://localhost:5000/api/health`。

---

## 3. 前端设置 (React + Vite)

前端负责数据的可视化展示。

### 步骤：
1. **进入项目根目录**（如果当前在 backend 目录，请先返回）：
   ```powershell
   cd ..
   ```

2. **安装 Node.js 依赖**：
   > [!IMPORTANT]
   > 由于 React 19 存在依赖版本冲突，请使用以下命令安装：
   ```powershell
   npm install --legacy-peer-deps
   ```

3. **启动开发服务器**：
   ```powershell
   npm run dev
   ```
   > [!TIP]
   > 前端默认运行在 `http://localhost:3000`。

---

## 4. 访问应用

1. 确保后端和前端都已启动。
2. 在浏览器中访问：**`http://localhost:3000`**。
3. 输入股票代码（如 `AAPL`）并开始分析。

---

## 5. 故障排除 (Troubleshooting)

- **npm : 无法加载文件...因为在此系统上禁止运行脚本** (PowerShell 报错):
  - 这是因为 PowerShell 的执行策略限制。请以**管理员身份**打开 PowerShell 并运行以下命令：
    ```powershell
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    ```
  - 输入 `Y` 确认更改，然后重新尝试运行 `npm install`。
- **CORS 错误**：确保后端已启动并运行在 5000 端口。后端代码已配置允许来自前端的跨域请求。
- **依赖安装失败**：
  - 如果 `pip install` 报错，请尝试更新 pip：`python -m pip install --upgrade pip`。
  - 如果 `npm install` 报错，请检查您的 Node.js 版本。
- **数据加载失败**：请检查您的网络连接，应用需要访问 Yahoo Finance API 获取数据。
