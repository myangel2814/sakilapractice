from flask import Flask
from flaskext.mysql import MySQL
import mysql.connector  # Thêm thư viện để xử lý lỗi kết nối
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np
import plotly.express as px

app = Flask(__name__)

# Kết nối database
def getConnect(server, port, database, username, password):
    try:
        mysql = MySQL()
        app.config['MYSQL_DATABASE_HOST'] = server
        app.config['MYSQL_DATABASE_PORT'] = port
        app.config['MYSQL_DATABASE_DB'] = database
        app.config['MYSQL_DATABASE_USER'] = username
        app.config['MYSQL_DATABASE_PASSWORD'] = password
        mysql.init_app(app)
        conn = mysql.connect()
        return conn
    except mysql.connector.Error as e:
        print("Error = ", e)
    return None

def closeConnection(conn):
    if conn is not None:
        conn.close()

# Truy vấn database
def queryDataset(conn, sql):
    cursor = conn.cursor()
    cursor.execute(sql)
    df = pd.DataFrame(cursor.fetchall(), columns=[col[0] for col in cursor.description])  # Lấy tên cột
    return df

conn = getConnect('localhost', 3306, 'sakila', 'root', 'My123!')

sql1 = "SELECT * FROM customer"
df1 = queryDataset(conn, sql1)
print(df1)

sql2 = """
SELECT 
    c.customer_id, 
    f.length, 
    i.inventory_id,    
    f.replacement_cost, 
    f.rating 
FROM customer c 
JOIN rental r ON c.customer_id = r.customer_id 
JOIN inventory i ON r.inventory_id = i.inventory_id 
JOIN film f ON i.film_id = f.film_id
"""

df2 = queryDataset(conn, sql2)
df2.columns = ['CustomerId', 'Length', 'InventoryID', 'Replacement_Cost', 'Rating']

# Chuyển đổi rating sang số
rating_mapping = {"G": 1, "PG": 2, "PG-13": 3, "R": 4, "NC-17": 5}
df2['Rating'] = df2['Rating'].map(rating_mapping)

# Nhóm dữ liệu theo CustomerId
df2 = df2.groupby('CustomerId').agg({
    'Length': 'sum',
    'InventoryID': 'nunique',
    'Replacement_Cost': 'mean',
    'Rating': 'mean'
})

# Vẽ biểu đồ histogram
def showHistogram(df, columns):
    num_columns = len(columns)  # Số cột cần vẽ histogram
    rows = (num_columns // 2) + (num_columns % 2)  # Chia thành hàng

    plt.figure(figsize=(10, 5 * rows))  # Điều chỉnh kích thước hình ảnh

    for i, column in enumerate(columns, 1):
        plt.subplot(rows, 2, i)  # Sử dụng 2 cột thay vì 3
        plt.subplots_adjust(hspace=0.5, wspace=0.3)
        sns.histplot(df[column], bins=32, kde=True)
        plt.title(f'Histogram of {column}')

    plt.show()

# Tìm số cụm bằng phương pháp Elbow
def elbowMethod(df, columnsForElbow):
    X = df.loc[:, columnsForElbow].values
    inertia = []

    for n in range(1, 11):
        model = KMeans(n_clusters=n, init='k-means++', max_iter=500, random_state=42)
        model.fit(X)
        inertia.append(model.inertia_)

    plt.figure(figsize=(15, 6))
    plt.plot(range(1, 11), inertia, 'o-')
    plt.xlabel('Số lượng cụm')
    plt.ylabel('Tổng bình phương khoảng cách cụm')
    plt.show()

columns = ["Length", "InventoryID", "Replacement_Cost", "Rating"]
elbowMethod(df2, columns)
# Chạy KMeans Clustering
def runKMeans(X, cluster):
    model = KMeans(n_clusters=cluster, init='k-means++', max_iter=500, random_state=42)
    model.fit(X)
    labels = model.labels_
    centroids = model.cluster_centers_
    y_kmeans = model.predict(X)
    return y_kmeans, centroids, labels

X = df2.loc[:, columns].values
cluster = 3
y_kmeans, centroids, labels = runKMeans(X, cluster)

df2["cluster"] = labels

# Vẽ biểu đồ 3D KMeans
def visualize3DKmeans(df, columns, hover_data, cluster):
    fig = px.scatter_3d(df, x=columns[0], y=columns[1], z=columns[2], color='cluster',
                        hover_data=hover_data, category_orders={"cluster": range(0, cluster)})
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()

hover_data = df2.columns
visualize3DKmeans(df2, columns, hover_data, cluster)

# Lọc từng cụm
cluster0 = df2[df2['cluster'] == 0]
cluster1 = df2[df2['cluster'] == 1]
cluster2 = df2[df2['cluster'] == 2]


