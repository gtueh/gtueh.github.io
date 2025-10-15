# --- Import các thư viện cần thiết ---
import pandas as pd
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns

# --- Bước 1: Tải và tiền xử lý dữ liệu ---

# Đọc file dữ liệu (đảm bảo file 'OnlineRetail.csv' nằm cùng thư mục)
df = pd.read_csv('OnlineRetail.csv', encoding='ISO-8859-1')

# 1. Loại bỏ các dòng không có mã khách hàng
df.dropna(subset=['CustomerID'], inplace=True)

# 2. Loại bỏ các giao dịch trả hàng (Quantity < 0)
df = df[df['Quantity'] > 0]

# 3. Chuyển CustomerID sang kiểu số nguyên
df['CustomerID'] = df['CustomerID'].astype(int)

# 4. Tạo cột Tổng tiền (TotalPrice)
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# 5. Chuyển cột InvoiceDate sang kiểu datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format="%d-%m-%Y %H:%M")

# --- Bước 2: Tính các đặc trưng RFM (Recency – Frequency – Monetary) ---

# Xác định ngày chụp (snapshot_date) là ngày giao dịch cuối cùng + 1 ngày
snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

# Gom nhóm theo CustomerID để tính Recency, Frequency và Monetary
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda date: (snapshot_date - date.max()).days,  # Recency
    'InvoiceNo': 'nunique',                                        # Frequency
    'TotalPrice': 'sum'                                            # Monetary
})

# Đổi tên các cột cho dễ hiểu
rfm.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalPrice': 'MonetaryValue'
}, inplace=True)

print(rfm.head())

# --- Bước 3: Chuẩn hóa dữ liệu ---
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# --- Bước 4: Xây dựng mô hình phân cụm đa cấp (Hierarchical Clustering) ---

# Tạo ma trận liên kết bằng phương pháp Ward
linked = linkage(rfm_scaled, method='ward')

# Vẽ biểu đồ Dendrogram
plt.figure(figsize=(14, 8))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True,
           truncate_mode='lastp',   # Hiển thị p cụm cuối cùng
           p=10                     # ✅ p=10 là giá trị tối ưu cho dataset này
          )
plt.title('Dendrogram Phân cụm Khách hàng 🌳', fontsize=16)
plt.xlabel('Số lượng điểm trong cụm', fontsize=13)
plt.ylabel('Khoảng cách (Ward)', fontsize=13)
plt.show()

# --- Bước 5: Xây dựng mô hình Agglomerative Clustering ---
agg_cluster = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg_cluster.fit_predict(rfm_scaled)

# Gán nhãn cụm vào bảng RFM
rfm['Cluster'] = labels

# --- Bước 6: Phân tích kết quả ---
cluster_summary = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': 'mean'
}).round(2)

print("\n📊 Tóm tắt đặc trưng trung bình của các cụm khách hàng:")
print(cluster_summary)

# --- Bước 7: Trực quan hóa kết quả ---
plt.figure(figsize=(10, 7))
sns.scatterplot(data=rfm, x='Frequency', y='MonetaryValue',
                hue='Cluster', palette='viridis', s=80)
plt.title('Phân cụm Khách hàng theo Tần suất và Giá trị chi tiêu 🛍️', fontsize=16)
plt.xlabel('Tần suất mua hàng (Frequency)', fontsize=12)
plt.ylabel('Tổng giá trị chi tiêu (Monetary)', fontsize=12)
plt.legend(title='Nhóm khách hàng')
plt.show()