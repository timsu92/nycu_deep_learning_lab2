# 專案介紹

本專案實作了兩種語義分割模型：UNet 與 ResNet34+UNet，使用 PyTorch 框架，並以 Oxford-IIIT Pet 資料集為例進行訓練與評估。
主要目標是對圖片進行精確的物件分割，區分前景與背景，並利用多種影像增強與訓練技巧提升模型效能，同時採用自動混合精度（AMP）來加速訓練並節省 VRAM。

# 主要功能與架構

* **訓練（train.py）**
  執行模型訓練流程，支援指定模型、批次大小、訓練輪數等參數。
  使用 RMSprop 優化器、CosineAnnealingLR 學習率調整器與 BCEWithLogitsLoss 與 Dice Loss 結合的損失函數。
  透過 AMP 進行自動混合精度加速，減少記憶體佔用並提升訓練速度。

* **推論（inference.py）**
  載入訓練好的模型檢查點，對測試資料進行推論並計算 Dice score。
  並以視覺化方式顯示結果圖片，帶有分割遮罩。支援批次大小調整與逐批觀看功能。

* **評估（evaluate.py）**
  提供計算整體資料集 Dice score 的功能，專注於模型效能評估。

# 執行指令範例

```bash
# 安裝依賴 (請依 CUDA 版本調整)
uv sync --frozen

# UNet 訓練
python3 -m src.train --data_path dataset/oxford-iiit-pet/ --epochs 20 --batch_size 34 --model unet --random-seed 42

# UNet 推論與驗證
python3 -m src.inference --batch_size 34 --model unet --checkpoint saved_models/UNet_20.pt

# ResNet34-UNet 訓練
python3 -m src.train --data_path dataset/oxford-iiit-pet/ --epochs 20 --batch_size 72 --model resnet34_unet --random-seed 100

# ResNet34-UNet 推論與驗證
python3 -m src.inference --batch_size 72 --model resnet34_unet --checkpoint saved_models/ResNet_20.pt
```

# 模型架構說明

* **UNet (`unet.py`, `libUnet.py`)**
  完整的 UNet 結構，包括：

  * DoubleConv：兩層卷積＋BatchNorm＋ReLU
  * Down：最大池化後接 DoubleConv
  * Up：跳接(skip connection)與上採樣結合
  * Out：最後的分類卷積層
    原始設計為單通道輸入，因應 RGB 圖片改為 3 通道。

* **ResNet34 + UNet (`resnet34_unet.py`, `libResnet34_unet.py`)**
  結合 ResNet34 作為編碼器，UNet 解碼器負責還原空間資訊。
  編碼器採用 ResNet34 除去最後平均池化層的結構，解碼器則自訂五個解碼區塊，並連接三個編碼器的跳接層（256、128、64 通道）。
