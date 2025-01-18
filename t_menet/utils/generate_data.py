import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

# 자유도 설정
df = 7

# t-분포 데이터 생성
n_sample = 1000

# ID 컬럼 생성 (0~9까지 랜덤하게 배정)
ids = np.random.choice(10, size=n_sample)  # 0~9 사이의 숫자를 랜덤 배정
x = np.linspace(-10, 10, n_sample)
y = t.pdf(x, df)

# 데이터프레임으로 변환
data = pd.DataFrame({"id": ids, "x": x, "y": y})

# CSV로 저장
csv_file_path = "/Users/dkmoon/Desktop/WorkSpace/DKU/t-distribution MeNet/data/t_distribution_data.csv"
data.to_csv(csv_file_path, index=False)

# # 그래프 그리기
# plt.plot(x, y, label=f"t-distribution (df={df})")
# plt.title("T-Distribution Simulation")
# plt.xlabel("x")
# plt.ylabel("Probability Density")
# plt.legend()
# plt.grid()
# plt.show()