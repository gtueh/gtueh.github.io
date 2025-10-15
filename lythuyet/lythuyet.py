# Import các thư viện cần thiết
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt

# 1️⃣ Đọc dữ liệu rượu vang
wine_df = pd.read_csv("wine-clustering.csv")

# 2️⃣ Xây dựng mô hình phân cụm đa cấp (Agglomerative Clustering)
ac = AgglomerativeClustering(
    n_clusters=3,        # số cụm cần tạo
    linkage='average'    # phương pháp gộp (ward, complete, average, single)
)

# 3️⃣ Huấn luyện mô hình và dự đoán cụm
ac_clusters = ac.fit_predict(wine_df)

# 4️⃣ Trực quan hóa kết quả phân cụm
plt.scatter(wine_df.values[:, 0], wine_df.values[:, 1], c=ac_clusters, cmap='rainbow')
plt.title("Wine Clusters from Agglomerative Clustering")
plt.xlabel("OD Reading")
plt.ylabel("Proline")
plt.show()

# 5️⃣ (Tùy chọn) Đánh giá độ tốt cụm bằng Silhouette Score
score = silhouette_score(wine_df, ac_clusters)
print("Silhouette Score:", score)
