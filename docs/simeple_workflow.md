# Training Process
1. EDA：
- 列出這份 datasets/train_2022.csv 的 columns，並且分析每個 column 的資料型態、缺失值、分佈情況等。最後列出這份資料集會有那些問題。跟EDA有關的程式碼與分析結果都放在 EDA/ 資料夾內部。
2. Data Splitting
- 將資料集分割為訓練集和驗證集，以評估模型的表現。最後將分割後的資料集儲存為新的 CSV 檔案，並且放在 Data Splitting/ 資料夾內部。
3. Data Preprocessing
- 根據 EDA 的結果，對資料集進行適當的清洗和處理，例如填補缺失值、刪除不必要的 columns、轉換資料型態等。最後將處理後的資料集儲存為新的 CSV 檔案，並且放在 Preprocessing/ 資料夾內部。
- 特殊標記先保留
- text 先統一大小寫，但保留標點符號
4. Model Training
- 使用處理後的資料集來訓練模型，選擇適合的演算法和參數，並且評估模型的表現。最後將訓練好的模型儲存為檔案，並且放在 Model/ 資料夾內部。

# Testing Process
1. Data Preprocessing
- 使用與訓練資料相同的方式對測試資料集進行處理，確保測試資料集與訓練資料集具有相同的格式和特徵。最後將處理後的測試資料集儲存為新的 CSV 檔案，並且放在 Preprocessing/ 資料夾內部。
2. Model Testing
- 使用訓練好的模型對測試資料集進行預測，並且評估模型的表現。最後將預測結果儲存為新的 CSV 檔案，並且放在 Testing/ 資料夾內部。