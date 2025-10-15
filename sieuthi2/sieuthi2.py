# Import các thư viện cần thiết
import pandas as pd
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns

# Tải dữ liệu từ file CSV
# Lưu ý: Đảm bảo file 'OnlineRetail.csv' nằm cùng thư mục với code của bạn
df = pd.read_csv('OnlineRetail.csv', encoding='ISO-8859-1')
# --- Làm sạch dữ liệu ---

# 1. Bỏ các dòng không có CustomerID
df.dropna(subset=['CustomerID'], inplace=True)

# 2. Loại bỏ các giao dịch trả hàng (có Quantity < 0)
df = df[df['Quantity'] > 0]

# 3. Chuyển đổi kiểu dữ liệu CustomerID sang số nguyên
df['CustomerID'] = df['CustomerID'].astype(int)

# 4. Tạo cột 'TotalPrice' = Quantity * UnitPrice
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# 5. Chuyển đổi cột 'InvoiceDate' sang kiểu datetime
# Ta cần chỉ định đúng định dạng ngày tháng của dữ liệu là Day-Month-Year
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format="%d-%m-%Y %H:%M")

print("Tiền xử lý dữ liệu hoàn tất.")
# --- Kỹ thuật đặc trưng RFM ---

# 1. Lấy ngày giao dịch cuối cùng trong dữ liệu làm mốc
snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

# 2. Gom nhóm theo CustomerID và tính các giá trị R, F, M
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda date: (snapshot_date - date.max()).days, # Recency
    'InvoiceNo': 'nunique',                                      # Frequency (đếm số hóa đơn duy nhất)
    'TotalPrice': 'sum'                                          # Monetary
})

# 3. Đổi tên các cột cho dễ hiểu
rfm.rename(columns={'InvoiceDate': 'Recency',
                    'InvoiceNo': 'Frequency',
                    'TotalPrice': 'MonetaryValue'}, inplace=True)

df_rfm = rfm
print("Đã tạo xong các đặc trưng RFM:")
print(df_rfm.head())

# 4. Chuẩn hóa dữ liệu RFM
# Vì các cột R, F, M có thang đo rất khác nhau, việc chuẩn hóa là bắt buộc
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(df_rfm)
# Tạo ma trận liên kết (linkage matrix) bằng phương pháp 'ward'
# 'ward' là một phương pháp phổ biến nhằm tối thiểu hóa phương sai trong mỗi cụm
linked = linkage(rfm_scaled, method='ward')

# Vẽ dendrogram
plt.figure(figsize=(14, 8))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True,
           truncate_mode='lastp', # Chỉ hiển thị p cụm cuối cùng được hợp nhất
           p=12 # p=12 giúp biểu đồ gọn hơn
          )
plt.title('Dendrogram Phân cụm Khách hàng 🌳', fontsize=16)
plt.xlabel('Số lượng điểm trong cụm', fontsize=14)
plt.ylabel('Khoảng cách Ward', fontsize=14)
plt.show()
# Xây dựng mô hình với 3 cụm
agg_cluster = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg_cluster.fit_predict(rfm_scaled)

# Gán nhãn cụm vừa tìm được vào lại DataFrame RFM
df_rfm['Cluster'] = labels
# Tính giá trị RFM trung bình cho mỗi cụm
cluster_summary = df_rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': 'mean'
}).round(2)

print("\n--- Đặc điểm trung bình của các cụm ---")
print(cluster_summary)

# Trực quan hóa các cụm bằng biểu đồ phân tán
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_rfm, x='Frequency', y='MonetaryValue', hue='Cluster', palette='viridis', s=80)
plt.title('Phân cụm Khách hàng theo Tần suất và Giá trị tiền hàng 🛍️', fontsize=16)
plt.xlabel('Tần suất (Số lần mua)', fontsize=12)
plt.ylabel('Giá trị tiền hàng (Tổng chi tiêu)', fontsize=12)
plt.legend(title='Nhóm Khách hàng')
plt.show()