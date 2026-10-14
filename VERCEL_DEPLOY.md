# Vercel 部署指南（50 Startups Regression Lab）

## 專案與部署方式

- Vite / Chart.js：Vercel 靜態輸出 `dist/`
- FastAPI / sklearn：Vercel Python Function，入口 `api/index.py`
- 同網域 API：`/api/health`, `/api/analyze`, `/api/compare`, `/api/feature-trends`, `/api/predict`
- 範例 CSV：`public/50_Startups.csv`，建置後可從 `/50_Startups.csv` 讀取
- 不使用資料庫、沒有持久化模型：每次分析由 Python Function 計算

## A. 上傳 GitHub

在 `startup-lab` 資料夾中執行：

```powershell
git init
git add .
git commit -m "feat: prepare 50 startups lab for vercel"
git branch -M main
git remote add origin https://github.com/<你的帳號>/<你的repo>.git
git push -u origin main
```

如果原本已有 Git 儲存庫，請跳過 init / remote add，直接 commit + push。

## B. 匯入 Vercel

1. 開啟 https://vercel.com/new ，從 GitHub Import 專案。
2. Root Directory 選擇包含 `vercel.json`、`package.json`、`api/` 的資料夾。若 GitHub 根目錄是 `startup-lab/`，設定 `startup-lab`；若直接上傳其內容，選 `./`。
3. Framework Preset: Vite；Build Command: `npm run build`；Output Directory: `dist`。
4. Node.js 建議 22；Python 版本由 `.python-version` 指定為 3.12。
5. 按 Deploy。不要將 `node_modules` 或 `.venv` 上傳。

## C. 驗收

- `https://<你的專案>.vercel.app/`：網頁
- `https://<你的專案>.vercel.app/50_Startups.csv`：範例資料
- `https://<你的專案>.vercel.app/api/health`：應回應 `{"status":"ok"}`
- `https://<你的專案>.vercel.app/api/docs`：依 Vercel 路由實際狀況檢查 API 文件
- 在網頁依序測試：執行模型分析、比較八種方法、繪製特徵數量趨勢、企業預測

如果首頁正常但 API 回傳 404，先查看 Vercel 的 Functions 和 Deployment Logs，確認 `api/index.py` 有被識別為 Python Function，再檢查 `vercel.json` 的 Rewrite。若 API 返回 500，查看 Python 的 import error 和套件安裝記錄。

## D. 本機測試

終端機 1：
```powershell
python -m pip install -r requirements.txt
python -m pip install uvicorn
python -m uvicorn api.index:app --reload --port 8000
```
終端機 2：
```powershell
npm.cmd install
npm.cmd run dev
```
開啟 http://localhost:5173 ， Vite 開發伺服器會代理 `/api` 到本機 8000 port。

## 注意事項

- 專案的 `/api/feature-trends` 會做多次交叉驗證，較大 CSV 可能超過 Serverless 執行時間，請先使用內建 50 筆資料測試。
- 上傳 CSV 會由瀏覽器轉成 JSON 傳送後端；部署平台有 request body 大小限制。
- Vercel Python Runtime 的套件大小、執行時間和方案限制可能改變，若部署失敗請以 Vercel Deployment / Function Logs 為準。
- 專案使用 `Chart.getChart(canvas)?.destroy()` 的方式修復 Chart.js Canvas 重用問題。
- 本專案已做本機程式測試，但未使用你的 Vercel 帳號進行線上部署，不能保證實際部署免除平台限制。
