import statistics
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

class KM:
    def __init__(self, data):
        """
        初始化聚类分析器。
        参数:
            data: dict, 包含 'days' 列表的多天流量数据。
        """
        self.data = data
        self.one_day_data = self.aggregate_daily_median()
        self.X = self.build_feature_matrix()

    # 1. 将多天数据聚合为一天代表数据（按小时取中位数）
    def aggregate_daily_median(self):
        days = self.data.get("days", None)
        flow_names = self.data.get("flow_names", None)
        num_hours = 24
        num_flows = len(flow_names)

        # 初始化收集器：hours_flows[h][f] 存储第 h 小时、第 f 个流向在多天中的值
        hours_flows = [[[] for _ in range(num_flows)] for _ in range(num_hours)]

        # 遍历每一天、每一小时，收集流量值
        for day in days:
            for hour_item in day['hours']:
                h = hour_item["hour"]
                flows = hour_item["flows"]  # 修正：从 hour_item 中取 flows
                # 校验：每小时流向数必须与 flow_names 长度一致，防止 8 维数据错位
                if len(flows) != num_flows:
                    raise ValueError(
                        f"{day.get('day', '?')} 第 {h} 时的 flows 长度 "
                        f"{len(flows)} 与 flow_names 长度 {num_flows} 不一致: {flows}"
                    )
                for f in range(num_flows):
                    hours_flows[h][f].append(flows[f])

        # 对每个小时、每个流向计算中位数
        median_hours = []
        for h in range(num_hours):
            median_flows = []
            for f in range(num_flows):
                median_val = statistics.median(hours_flows[h][f])
                median_flows.append(median_val)
            # 注意：median_hours.append 应在内层循环结束后执行
            median_hours.append({
                "hour": h,
                "flows": median_flows
            })

        aggregated = {
            "intersection_id": self.data.get("intersection_id", ""),
            "flow_names": flow_names,
            "hours": median_hours
        }
        return aggregated

    # 2. 构建特征矩阵 X (24, n)
    def build_feature_matrix(self):
        hours = self.one_day_data.get("hours")
        X = np.array([hour['flows'] for hour in hours])
        return X

    # 3. 数据标准化（Z-score）
    def standardize(self):
        mean = np.mean(self.X, axis=0)
        std = np.std(self.X, axis=0)
        # 防止除以零
        std[std == 0] = 1
        Z = (self.X - mean) / std
        return Z, mean, std

    # 4. 通过轮廓系数选择最佳聚类数 K
    def find_optimal_k(self, Z, k_range=range(4, 9)):
        scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
            labels = kmeans.fit_predict(Z)
            score = silhouette_score(Z, labels)
            scores.append(score)
        best_k = k_range[np.argmax(scores)]
        return best_k, scores

    # 5. 执行 K-means 聚类
    def kmeans_cluster(self, Z, k):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = kmeans.fit_predict(Z)
        return labels, kmeans

    # 6. 时段连续性后处理
    @staticmethod
    def postprocess_labels(labels, min_duration=1):
        """
        对逐小时聚类标签进行时间连续性后处理。

        参数:
            labels: list 或 np.ndarray, 长度 24，每个元素为小时对应的类别标签。
            min_duration: 最短时段长度（小时），小于该长度的孤立时段将被合并。

        返回:
            segments: list of dict, 每个元素包含 'start_hour', 'end_hour', 'label'。
                      end_hour 为不包含的小时（即 [start, end) 区间）。
        """
        labels = list(labels)
        # 合并相邻同类
        segments = []
        current_label = labels[0]
        start = 0
        for i in range(1, len(labels)):
            if labels[i] != current_label:
                segments.append({'start_hour': start, 'end_hour': i, 'label': current_label})
                start = i
                current_label = labels[i]
        segments.append({'start_hour': start, 'end_hour': len(labels), 'label': current_label})

        # 处理孤立短时段（长度 < min_duration 的段）
        processed = []
        for seg in segments:
            dur = seg['end_hour'] - seg['start_hour']
            if dur >= min_duration:
                processed.append(seg)
            else:
                # 孤立段合并到前一个段（若存在）
                if processed:
                    prev = processed[-1]
                    prev['end_hour'] = seg['end_hour']
                else:
                    # 若第一段就过短，暂时保留，等待后续处理
                    processed.append(seg)

        # 处理跨零点情况：若首尾标签相同，合并首尾两段
        if len(processed) >= 2 and processed[0]['label'] == processed[-1]['label']:
            # 将最后一段合并到第一段，并调整起始小时
            first = processed[0]
            last = processed[-1]
            # 假设 last 是跨零点后的延续，其 start_hour 实际上应小于 24，
            # 这里将 last 的开始时间视为前一日延续，故将 first 的 start_hour 前移
            first['start_hour'] = last['start_hour'] - 24  # 用负数表示跨日
            # 删除最后一段
            processed.pop()
        return processed

    # 7. 完整流程：聚合 -> 标准化 -> 选K -> 聚类 -> 后处理
    def run(self):
        Z, mean, std = self.standardize()
        best_k, scores = self.find_optimal_k(Z)
        labels, kmeans = self.kmeans_cluster(Z, best_k)
        segments = self.postprocess_labels(labels)
        return {
            'best_k': best_k,
            'scores': scores,
            'labels': labels,
            'segments': segments,
            'standardized_data': Z,
            'mean': mean,
            'std': std
        }

if __name__ =="__main__":
    with open("数据.json","r",encoding='utf-8') as f:
        data = json.load(f)
    km = KM(data)
    result = km.run()
    print("最佳聚类数:", result['best_k'])
    print("逐小时标签:", result['labels'])
    print("控制时段:")
    for seg in result['segments']:
        print(f"  {seg['start_hour']:02d}:00 - {seg['end_hour']:02d}:00, 类别 {seg['label']}")

