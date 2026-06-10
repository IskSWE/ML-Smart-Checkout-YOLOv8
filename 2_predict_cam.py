import gradio as gr
import cv2
from ultralytics import YOLO
import numpy as np

# 載入模型權重（優先讀取 best.pt，若不存在則直接調用官方輕量化 yolov8n.pt）
try:
    model = YOLO("best.pt")
except:
    model = YOLO("yolov8n.pt")

# 定義無人商店智能 POS 機之商品價格字典 (以新台幣 TWD 計價)
PRICE_TAG = {
    "cell phone": 15000,
    "laptop": 35000,
    "keyboard": 1200,
    "mouse": 600,
    "cup": 150,
    "bottle": 30
}

def smart_checkout(image, conf_threshold):
    """物件偵測與智能結帳計價之機器學習核心演算法"""
    if image is None:
        return None, "請上傳商品影像以進行結帳。"
    
    # 執行 YOLOv8 Anchor-Free 單階段多目標偵測
    results = model(image, conf=conf_threshold)
    annotated_frame = results.plot()
    
    detected_objects = {}
    total_amount = 0
    
    # 遍歷所有預測出的邊界框 (Bounding Boxes)
    boxes = results.boxes
    for box in boxes:
        cls_id = int(box.cls)
        cls_name = model.names[cls_id]
        
        # 統計商品數量
        detected_objects[cls_name] = detected_objects.get(cls_name, 0) + 1
        
        # 若商品存在於店家字典中，自動加總總金額
        if cls_name in PRICE_TAG:
            total_amount += PRICE_TAG[cls_name]
            
    # 格式化輸出前台智能 POS 機的結帳收據報告
    summary_text = "========= 🛒 智慧自助結帳明細 =========\n"
    if not detected_objects:
        summary_text += "未偵測到任何商品。請降低信心度閾值並重新放置物品。\n"
    else:
        for obj, count in detected_objects.items():
            price = PRICE_TAG.get(obj, "未上架商品")
            summary_text += f"• {obj} x {count} (單價: {price})\n"
            
    summary_text += "---------------------------------------\n"
    summary_text += f"💰 本次結帳總金額: ${total_amount} TWD\n"
    summary_text += "======================================="
    
    return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), summary_text

# 構建符合期末成果審查之 Gradio 互動式 Web 前端介面
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛒 機器學習期末專題：無人商店智慧自助結帳與邊緣運算管理系統")
    gr.Markdown("### 學生姓名：邱新舜 | 技術棧：YOLOv8-Nano + Gradio + OpenCV")
    
    with gr.Row():
        # 左側控制區 (輸入端)
        with gr.Column(scale=1):
            input_img = gr.Image(type="numpy", label="📥 放入結帳商品 (上傳測試影像)")
            conf_slider = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.25, step=0.05, 
                label="🎛️ 機器學習信心度閾值 (Confidence Threshold)"
            )
            btn = gr.Button("🚀 掃描商品並計算金額", variant="primary")
            
        # 右側結果區 (輸出端)
        with gr.Column(scale=1):
            output_img = gr.Image(label="🖼️ 商品邊界框定位結果 (Bounding Box)")
            output_txt = gr.Textbox(label="📊 智能 POS 機結帳收據", placeholder="等待結帳...")
            
    # 綁定按鈕點擊事件，進行前後端資料流串接
    btn.click(fn=smart_checkout, inputs=[input_img, conf_slider], outputs=[output_img, output_txt])

if __name__ == "__main__":
    demo.launch(share=False)
