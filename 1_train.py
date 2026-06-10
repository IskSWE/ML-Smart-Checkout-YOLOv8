from ultralytics import YOLO

# 載入輕量化模型權重，並使用指定的 data.yaml 設定檔進行微調
model = YOLO("yolov8n.pt")

if __name__ == "__main__":
    # 執行模型微調訓練策略，設定為 50 輪 (Epochs)
    model.train(data="data.yaml", epochs=50, imgsz=640)
