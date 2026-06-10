# 無人商店智慧自助結帳與邊緣運算管理系統 (ML-Smart-Checkout-YOLOv8)

機器學習期末專題報告
- **學生姓名**：邱新舜
- **課程名稱**：機器學習

## 🚀 專案簡介
本專題基於最新一代單階段（One-Stage）輕量化物件偵測框架 **YOLOv8-Nano** 結合 **Gradio 網頁前端**，實現了一套部署於純 CPU 邊緣端硬體的即時零售商品自動計價與多目標影像分析原型系統。

## 📁 專案結構
- `data.yaml`: 定義測試資料集路徑與商品類別映射。
- `requirements.txt`: 專案相依套件清單。
- `1_train.py`: 遷移學習微調模型之常規訓練腳本。
- `2_predict_cam.py`: 核心 DEMO 主程式（Gradio 互動式自助結帳 POS 系統）。

## ⚙️ 2步快速測試執行
1. 安裝必要機器學習與前端套件：
   ```bash
   pip install -r requirements.txt
   ```
2. 啟動 Gradio 本地網頁伺服器：
   ```bash
   python 2_predict_cam.py
   ```

