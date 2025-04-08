# -*- coding: utf-8 -*-
!pip install snownlp
!pip install gensim
!pip install --upgrade numpy
!pip install --upgrade gensim
!pip install stopwordsiso

import csv
import pandas as pd
from datetime import datetime

# Define the input and output file paths
input_file = '/content/D5.csv'        # Replace with the actual input file path
output_file = 'Dataset5_final.csv'    # Replace with the desired output file path

# Step 1: Read the input CSV file and add the "label" column
with open(input_file, 'r', newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    rows = list(reader)

if rows:
    # Add the "label" column in the header
    header = rows[0]
    header.append("label")

    # Locate the index for 'Status' and 'label'
    try:
        status_idx = header.index("Status")
        label_idx = header.index("label")
    except ValueError:
        print("CSV文件中必须包含'status'和'label'列")
        exit()

    # Process the data rows
    for row in rows[1:]:
        # Ensure the row is long enough
        if len(row) < len(header):
            row.extend([''] * (len(header) - len(row)))  # Extend the row to match header length

        # Standardize and process the 'Status' and 'label' columns
        status_val = row[status_idx].strip()

        if status_val == "1":
            row[label_idx] = "1"
            row[status_idx] = "1"
        else:
            row[label_idx] = "0"
            row[status_idx] = "0"

# Step 2: Write the updated content to a new CSV file
with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(rows)

print(f"处理完成，最终文件已输出为 {output_file}")

# Step 3: Process the saved CSV file using pandas
df = pd.read_csv(output_file, encoding='utf-8')

# 0. Data filtering
df = df[df['Type'] == 1]

# 1. Data processing: generating status column
df['label'] = df['label'].apply(lambda x: 1 if x == 1 else 0)

# 2. Time feature processing
df['timestamp'] = pd.to_datetime(df['CreateTime'], unit='s')
df['time_diff'] = df['timestamp'].diff().dt.total_seconds().fillna(0)

# 3. Text feature processing
df['message'] = df['StrContent'].fillna('').str.replace(r'<.*?>', '', regex=True)

# 4. User feature processing
user_mapping = {user: idx for idx, user in enumerate(df['Sender'].unique())}
df['user_id'] = df['Sender'].map(user_mapping)

# Keep only relevant columns
processed_df = df[['timestamp', 'user_id', 'message', 'time_diff', 'label']]
processed_df.reset_index(drop=True, inplace=True)

# Display the processed data
print(processed_df.head())

# Save the processed data to a new CSV file
processed_df.to_csv("processed_chat_data_5.csv", index=False, encoding="utf-8")
print("处理后的数据已保存为 processed_chat_data_5.csv")

from transformers import DistilBertTokenizer, DistilBertModel
import torch

# 检查 GPU 是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载 DistilBERT 模型和 Tokenizer，并将模型移动到 GPU
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

# 将消息转化为 DistilBERT 嵌入向量
def message_to_distilbert_embedding(messages):
    """
    将消息转化为 DistilBERT 嵌入向量
    Args:
        messages (list[str]): 消息文本列表
    Returns:
        torch.Tensor: 消息的平均嵌入向量
    """
    # 使用 GPU 处理 Tokenizer 输入
    inputs = tokenizer(messages, padding=True, truncation=True, return_tensors="pt").to(device)
    # 使用 GPU 运行模型
    outputs = model(**inputs)
    # 将隐藏状态的最后一层平均池化
    return outputs.last_hidden_state.mean(dim=1).cpu().detach().numpy()  # 返回到 CPU 进行存储

# 批量处理消息嵌入
batch_size = 32
message_embeddings = []
for i in range(0, len(processed_df), batch_size):
    batch_messages = processed_df['message'].iloc[i:i+batch_size].tolist()
    batch_embeddings = message_to_distilbert_embedding(batch_messages)
    message_embeddings.extend(batch_embeddings)

# 添加到 DataFrame
processed_df['message_embedding'] = message_embeddings
import numpy as np
processed_df['log_time_diff'] = np.log1p(processed_df['time_diff'])

import torch.nn as nn

user_embedding_layer = nn.Embedding(num_embeddings=len(user_mapping), embedding_dim=100)
processed_df['user_embedding'] = processed_df['user_id'].apply(
    lambda x: user_embedding_layer(torch.tensor(x))
)

def prepare_tcn_input(data):
    embeddings = torch.stack([torch.tensor(e) for e in data['message_embedding'].values.tolist()])
    time_features = torch.tensor(data['log_time_diff'].values).unsqueeze(1)
    user_features = torch.stack([torch.tensor(u) for u in data['user_embedding'].values.tolist()])
    return torch.cat([embeddings, time_features, user_features], dim=1)

tcn_inputs = prepare_tcn_input(processed_df)
tcn_inputs = prepare_tcn_input(processed_df).float()
print(tcn_inputs.shape)  # 应输出 [batch_size, feature_dim]

import torch
import torch.nn as nn
import torch.nn.functional as F

# 局部注意力层，用于计算消息上下文依赖性
class LocalAttention(nn.Module):
    def __init__(self, hidden_size, num_neighbors=5):
        super(LocalAttention, self).__init__()
        self.num_neighbors = num_neighbors
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        batch_size, seq_length, hidden_size = hidden_states.shape
        outputs = []
        for t in range(seq_length):
            query = self.W_q(hidden_states[:, t, :])
            start_idx = max(0, t - self.num_neighbors)
            context = hidden_states[:, start_idx:t+1, :]
            keys = self.W_k(context)
            scores = torch.bmm(query.unsqueeze(1), keys.permute(0, 2, 1)).squeeze(1)
            attention_weights = F.softmax(scores, dim=1)
            values = self.W_v(context)
            attended_representation = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)
            outputs.append(attended_representation)
        return torch.stack(outputs, dim=1)

# TCN + Local Attention 模型，输出一个固定向量（例如用于后续任务）
class TCNWithLocalAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_channels, kernel_size=3, num_neighbors=5):
        """
        参数说明：
          input_size   : 拼接后特征的维度，例如 embedding_dim + 1 + user_dim（例如 300 + 1 + 100 = 401）
          hidden_size  : 经过全连接层后得到的向量维度，也是最终输出向量的维度（例如 128）
          num_channels : TCN 层的输出通道数（例如 256）
          kernel_size  : 卷积核大小
          num_neighbors: 局部注意力层中考虑的邻居数量
        """
        super(TCNWithLocalAttention, self).__init__()
        self.tcn = nn.Conv1d(input_size, num_channels, kernel_size, padding=kernel_size//2)
        self.fc = nn.Linear(num_channels, hidden_size)
        self.local_attention = LocalAttention(hidden_size, num_neighbors)

    def forward(self, x):
        """
        参数:
          x: 输入 tensor，可以是二维 [batch_size, input_size]（自动扩展为 [batch_size, input_size, 1]）
             或三维 [batch_size, input_size, sequence_length]
        返回:
          vector_output: 每个样本的向量表示，形状为 [batch_size, hidden_size]
        """
        # 如果输入 x 是二维，则在最后一维扩展为 1（代表单一时间步）
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # 变为 [batch_size, input_size, 1]
        # 经过 TCN 层，输出形状为 [batch_size, num_channels, sequence_length]
        x = self.tcn(x)
        x = torch.relu(x)
        # 转置为 [batch_size, sequence_length, num_channels] 以便全连接层处理
        hidden_states = self.fc(x.permute(0, 2, 1))
        # 通过局部注意力获得上下文信息，输出形状为 [batch_size, sequence_length, hidden_size]
        attended_hidden_states = self.local_attention(hidden_states)
        # 这里采用取最后一个时间步的输出作为向量（你也可以改为全局池化）
        vector_output = attended_hidden_states[:, -1, :]
        return vector_output

import torch.optim as optim

# 不再依赖原来的 embedding_dim/user_dim 定义，直接使用数据中的特征数
input_size = 869
hidden_size = 128     # 最终输出向量的维度
num_channels = 256
epochs = 10

# 初始化 TCN 模型
model = TCNWithLocalAttention(input_size=tcn_inputs.shape[1], hidden_size=128, num_channels=256)

# 如果你的下游任务是分类任务（例如有10个类别），你可能需要一个分类层
num_classes = 2
classifier = nn.Linear(hidden_size, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=0.001)

# 确保 targets 是一个 LongTensor，形状为 [batch_size]
# 例如：targets = torch.randint(0, num_classes, (tcn_inputs.shape[0],))
# 请根据你的任务准备合适的 targets
# 这里仅为示例
targets = torch.randint(0, num_classes, (tcn_inputs.shape[0],))

# 训练循环
for epoch in range(epochs):
    optimizer.zero_grad()
    # 模型接收 tcn_inputs，如果输入为二维，内部会自动扩展为 [batch_size, input_size, 1]
    vector_output = model(tcn_inputs)  # 输出形状为 [batch_size, hidden_size]
    logits = classifier(vector_output)  # 将向量映射到类别，输出形状为 [batch_size, num_classes]
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# 训练结束后
model.eval()  # 切换到评估模式

with torch.no_grad():
    # tcn_inputs 为你的预处理输入数据，模型会自动判断输入维度（二维或三维）
    vector_output = model(tcn_inputs)  # 输出形状为 [batch_size, hidden_size]
    print("输出向量的 shape:", vector_output.shape)
    print("部分向量内容：", vector_output[:5])  # 打印前5个样本的向量

# 如果你想保存这些向量到文件，可以使用 numpy 保存为 .npy 文件
import numpy as np
# 如果模型在 GPU 上，请先将其转到 CPU
vector_np = vector_output.cpu().numpy()
np.save("output_vectors.npy", vector_np)
print("向量已经保存到 output_vectors.npy")
# 或者使用 Pandas 保存为 CSV 文件，这种方法可以自动处理行号等问题
df = pd.DataFrame(vector_np)
df.to_csv("output_vectors_pandas.csv", index=False)
print("向量已保存到 output_vectors_pandas.csv")

import pandas as pd
import numpy as np
import string
import torch
#from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#启发式特征读取
# 读取数据
file_path = "processed_chat_data_5.csv"
df = pd.read_csv(file_path)

# 疑问关键词分类（所有类别）
chinese_keywords_by_character = {
    "哪": ["哪一天", "哪个", "哪个国家", "哪个城市", "哪个地方", "哪个区域",
           "哪一位", "哪一类", "哪一种", "哪一款", "哪一家", "哪里", "哪儿", "哪些"],
    "何": ["何时", "何处", "何", "因何", "如何", "为何", "如何做",
           "如何解决", "怎么回事", "为什么", "为何如此", "为什么这样"],
    "怎": ["怎么", "怎么办", "怎么做", "怎样", "怎么样", "怎样做",
           "怎么解决", "怎么回事", "怎么样"],
    "多": ["多", "多么", "多大", "多高", "多长", "多宽", "多深",
           "多重", "多快", "多慢", "多贵", "多便宜", "多难", "多容易",
           "多少", "多久", "多长时间"],
    "什": ["什么", "什么时候", "什么意思", "什么情况",
           "什么问题", "什么原因", "什么结果", "干什么"],
    "几": ["几", "几时", "几个", "几件", "几次", "几天",
           "几年", "几个月", "几小时", "几分钟"],
    "是": ["是否", "是不是"],
    "能": ["能否", "能不能"],
    "可": ["可否", "可不可以"],
    "谁": ["谁", "哪位", "什么人"],
    "行": ["行不行"],
    "好": ["好不好"],
    "对": ["对不对"]
}

# 计算 keyword_overlap
def calculate_keyword_overlap(messages, keyword_dict):
    keyword_overlap_scores = []
    contains_question_mark = []

    for msg in messages:
        total_count = sum(1 for key, words in keyword_dict.items() for word in words if word in msg)

        # 计算 keyword_overlap（防止除零）
        score = total_count / len(msg) if len(msg) > 0 else 0
        keyword_overlap_scores.append(score)

        # 计算 contains_question_mark
        contains_question_mark.append(1 if "？" in msg else 0)

    return keyword_overlap_scores, contains_question_mark

# 计算 keyword_overlap 和 contains_question_mark
df['keyword_overlap'], df['contains_question_mark'] = calculate_keyword_overlap(df['message'], chinese_keywords_by_character)

# 计算消息长度
df['message_length'] = df['message'].apply(len)

# 计算标点符号比例
df['punctuation_ratio'] = df['message'].apply(lambda x: sum(1 for c in x if c in string.punctuation) / len(x) if len(x) > 0 else 0)

# 计算时间间隔归一化
df['log_time_diff'] = np.log1p(df['time_diff'])  # log(1 + time_diff)

# 是否长时间未回复（如超过 5 分钟）
df['long_silence'] = (df['time_diff'] > 300).astype(int)  # 1 表示长时间未回复

from snownlp import SnowNLP

# 使用 SnowNLP 计算情感得分
def sentiment_score_snownlp(messages):
    return [SnowNLP(msg).sentiments for msg in messages]

# 更新情感得分
df['sentiment_score'] = sentiment_score_snownlp(df['message'])

# 计算情感变化
df['sentiment_shift'] = abs(df['sentiment_score'].diff().fillna(0))

from sklearn.metrics.pairwise import cosine_similarity

# 注意：这里假设你的消息嵌入存储在 processed_df 中
# 如果消息嵌入实际上存放在 df 中，请改为 df['message_embedding']
embeddings = torch.stack([torch.tensor(e, dtype=torch.float32) for e in processed_df['message_embedding'].tolist()])
embeddings = embeddings.numpy()  # 转换为 numpy 数组

# 计算句子嵌入的余弦相似度
similarities = cosine_similarity(embeddings)
df['topic_deviation'] = [0] + [1 - similarities[i, i - 1] for i in range(1, len(similarities))]

# 选择启发式特征
heuristic_features = df[['keyword_overlap', 'contains_question_mark', 'message_length','punctuation_ratio',
                          'long_silence', 'sentiment_shift', 'topic_deviation']]

# 转换为 PyTorch 张量
heuristic_vector = torch.tensor(heuristic_features.values, dtype=torch.float32)

# 保存启发式特征为 .npy 文件
np.save("heuristic_features.npy", heuristic_vector.numpy())

print("启发式特征数据已保存为 heuristic_features.npy")


# 保存为 CSV
heuristic_features.to_csv("heuristic_features.csv", index=False, encoding="utf-8")
print("启发式特征数据已保存为 heuristic_features.csv")

import numpy as np
from sklearn.preprocessing import StandardScaler

# 加载 .npy 文件
heuristic_features = np.load("heuristic_features.npy")  # 形状: (n_samples, heuristic_dim)
output_vectors = np.load("output_vectors.npy")          # 形状: (n_samples, 128) 假设这是语义向量

# 使用 StandardScaler 对启发式特征进行标准化
scaler_heuristic = StandardScaler()
heuristic_features_standardized = scaler_heuristic.fit_transform(heuristic_features)

# 使用 StandardScaler 对语义向量进行标准化
scaler_output = StandardScaler()
output_vectors_standardized = scaler_output.fit_transform(output_vectors)

# 示例：将两个标准化后的特征按列拼接
combined_features = np.hstack((output_vectors_standardized, heuristic_features_standardized))
print("Combined Features Shape:", combined_features.shape)

# 保存拼接后的特征
np.save("combined_features_standardized.npy", combined_features)
print("整合后的标准化特征已保存为 combined_features_standardized.npy")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 加载整合后的标准化特征
print("向量已经保存到 combined_features_standardized.npy")
combined_features = np.load("combined_features_standardized.npy")

# 检查数据形状
print("Combined Features Shape:", combined_features.shape)

# 使用 KMeans 聚类
n_clusters = 2  # 假设分为两类：话题转移和非话题转移
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(combined_features)

# 评估聚类效果
silhouette_avg = silhouette_score(combined_features, clusters)
print(f"Silhouette Score for KMeans: {silhouette_avg:.4f}")

# 计算聚类结果比例
cluster_counts = np.bincount(clusters)
total_samples = len(clusters)
cluster_ratios = cluster_counts / total_samples

# 显示各类的比例
for i, ratio in enumerate(cluster_ratios):
    print(f"Cluster {i} Ratio: {ratio:.4%}")

# 使用 PCA 降维到 2D 以便可视化
pca = PCA(n_components=2)
combined_features_reduced = pca.fit_transform(combined_features)

plt.figure(figsize=(8, 6))
plt.scatter(combined_features_reduced[:, 0], combined_features_reduced[:, 1], c=clusters, cmap='viridis', s=5)
plt.title("Topic Shift Clusters (KMeans on Combined Standardized Features)")
plt.colorbar(label="Cluster")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.show()

# 加载原始数据
original_data = pd.read_csv("processed_chat_data_5.csv")
original_data['cluster_label'] = clusters

# 保存聚类结果到文件
output_csv_path = "topic_shift_clusters_kmeans.csv"
original_data.to_csv(output_csv_path, index=False, encoding="utf-8")
print(f"聚类结果已保存为 {output_csv_path}")

import csv

input_file = 'topic_shift_clusters_kmeans.csv'      # 输入文件
output_file = 'new_topic_shift_clusters_kmeans.csv'  # 输出新文件

with open(input_file, 'r', newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    rows = list(reader)

# 假设文件第一行为标题，且 cluster_label 在最后一列
if rows:
    header = rows[0]
    # 如果需要确保操作的是 cluster_label 列，也可检查 header[-1]
    if header[-1] != 'cluster_label':
        print("警告：最后一列的名称不是 'cluster_label'")
    rows[0] = header

    # 遍历数据行，交换 0 与 1
    for i in range(1, len(rows)):
        # 获取最后一列的值
        val = rows[i][-1].strip()
        # 如果值为 '1'，则修改为 '0'，反之如果为 '0'，则修改为 '1'
        if val == '1':
            rows[i][-1] = '0'
        elif val == '0':
            rows[i][-1] = '1'
        # 其他值保持不变

# 写入新的 CSV 文件
with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(rows)

print(f"处理完成，新的文件已输出为 {output_file}")

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import re
import string
import torch
from transformers import AutoTokenizer, AutoModel
import stopwordsiso as stopwords

# 加载数据
data_path = "processed_chat_data_5.csv"
df = pd.read_csv(data_path)

texts = df['message'].astype(str).tolist()

# 加载中文停用词
stopwords_zh = stopwords.stopwords('zh')

# SnowNLP分词安全版（严格过滤空串或异常串）
def preprocess_snownlp(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", "", text)  # 去除所有空白符
    text = re.sub(f"[{string.punctuation}]", "", text)
    if len(text.strip()) < 1:
        return ""
    try:
        words = SnowNLP(text).words
        words = [w for w in words if w.strip() and w not in stopwords_zh and len(w.strip()) > 1]
        return ' '.join(words)
    except:
        return ""

# 处理文本，并去除分词后为空的文本
processed_texts = [preprocess_snownlp(text) for text in texts]

# 只保留有效分词结果
valid_indices = [i for i, text in enumerate(processed_texts) if text.strip()]
processed_texts = [processed_texts[i] for i in valid_indices]
df_valid = df.iloc[valid_indices].reset_index(drop=True)

# --- 方法一：TF-IDF聚类 ---
tfidf_vectorizer = TfidfVectorizer(max_features=50)
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_texts)
labels_tfidf = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(tfidf_matrix)
df_valid['tfidf_cluster'] = labels_tfidf

# --- 方法二：LDA聚类 ---
lda_vectorizer = TfidfVectorizer(max_features=50)
lda_tfidf_matrix = lda_vectorizer.fit_transform(processed_texts)
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda_matrix = lda.fit_transform(lda_tfidf_matrix)
labels_lda = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(lda_matrix)
df_valid['lda_cluster'] = labels_lda

# --- 方法三：Word2Vec 监督学习 ---
# 使用Word2Vec生成文本特征
tokenized_texts = [text.split() for text in processed_texts]
w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=2, workers=4)

# 获取Word2Vec文档向量（通过取每个词的词向量的平均值）
def get_w2v_vector(text):
    vectors = [w2v_model.wv[word] for word in text if word in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v_model.vector_size)

w2v_vectors = np.array([get_w2v_vector(text) for text in tokenized_texts])

# 使用SVM进行监督学习
clf = SVC()  # 你可以选择其他分类器，如逻辑回归、随机森林等
clf.fit(w2v_vectors, df_valid['label'])  # 使用标签数据进行训练

# 预测并评估
predictions_w2v = clf.predict(w2v_vectors)
accuracy_w2v = accuracy_score(df_valid['label'], predictions_w2v)
df_valid['w2v_pred'] = predictions_w2v
print(f'Word2Vec Accuracy: {accuracy_w2v}')

# --- 方法四：DistilBERT聚类 ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')
model = AutoModel.from_pretrained('distilbert-base-multilingual-cased').to(device)

def get_distilbert_embedding(text):
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embeddings.squeeze()

# 使用DistilBERT生成文本特征
distilbert_embeddings = np.array([get_distilbert_embedding(text) for text in df_valid['message'].astype(str).tolist()])

# 使用KMeans聚类（无监督）
labels_distilbert = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(distilbert_embeddings)
df_valid['distilbert_cluster'] = labels_distilbert


# 存储最终结果
df_valid.to_csv('chat_data_all_clusters.csv', index=False)

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import numpy as np

# WindowDiff计算函数
def windowdiff(seg1, seg2, k):
    n = len(seg1)
    wd = 0
    for i in range(n - k):
        wd += abs(sum(seg1[i:i+k]) - sum(seg2[i:i+k])) > 0
    return wd / (n - k)

# 加载之前聚类结果的数据
df_clusters = pd.read_csv('chat_data_all_clusters.csv')
df_clusters = df_clusters.dropna(subset=['label', 'tfidf_cluster', 'lda_cluster', 'w2v_pred', 'distilbert_cluster'])

# 加载新上传的数据（作为对比）
df_compare = pd.read_csv('topic_shift_clusters_kmeans.csv')
df_compare = df_compare.dropna(subset=['label', 'cluster_label'])

# 确保数据长度一致（截断至最短的长度）
min_len = min(len(df_clusters), len(df_compare))
df_clusters = df_clusters.iloc[:min_len]
df_compare = df_compare.iloc[:min_len]

# 真实标签
true_labels = df_clusters['label'].astype(int).values

# 各方法的预测标签
predictions = {
    'TFIDF': df_clusters['tfidf_cluster'].astype(int).values,
    'LDA': df_clusters['lda_cluster'].astype(int).values,
    'Word2Vec': df_clusters['w2v_pred'].astype(int).values,
    'DistilBERT': df_clusters['distilbert_cluster'].astype(int).values,
    'TopicShift_KMeans': df_compare['cluster_label'].astype(int).values
}

# 计算窗口大小 k
k = int(round(np.sqrt(len(true_labels))/2))

# 计算并存储所有方法的结果
results = []

for method, pred_labels in predictions.items():
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='macro')  # 使用macro平均来处理多类别问题
    recall = recall_score(true_labels, pred_labels, average='macro')  # 计算宏平均的召回率
    precision = precision_score(true_labels, pred_labels, average='macro')  # 计算宏平均的精确率
    wd_score = windowdiff(true_labels, pred_labels, k)
    results.append({
        'Method': method,
        'Accuracy': accuracy,
        'F1 Score': f1,  # 添加F1分数
        'Recall': recall,  # 添加召回率
        'Precision': precision,  # 添加精确率
        'WindowDiff': wd_score
    })

# 展示对比结果
results_df = pd.DataFrame(results)
print("📊 对比实验结果：")
print(results_df)
# 输出到 txt 文件
with open("experiment_results_D5.txt", "w") as f:
    f.write("📊 对比实验结果：\n")
    f.write(results_df.to_string(index=False))

print("实验结果已保存到 'experiment_results_D5.txt'")