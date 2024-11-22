# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 생성 함수 정의
def generate_t_distribution_data(n_samples, df):
    """
    t-분포를 기반으로 x, y 데이터를 생성합니다.

    Args:
        n_samples (int): 생성할 데이터 개수
        df (int): t-분포의 자유도 (degrees of freedom)

    Returns:
        x (numpy.ndarray): t-분포에서 생성된 x 값
        y (numpy.ndarray): t-분포에서 생성된 y 값
    """
    x = np.random.standard_t(df, size=n_samples)  # t-분포 기반으로 x 생성
    y = np.random.standard_t(df, size=n_samples)  # t-분포 기반으로 y 생성
    return x, y

# 메인 실행 코드
if __name__ == "__main__":
    # 생성할 데이터의 개수와 자유도 설정
    n_samples = 10000  # 생성할 데이터 개수
    degrees_of_freedom = 7  # t-분포의 자유도

    # 데이터 생성
    x, y = generate_t_distribution_data(n_samples, degrees_of_freedom)

    # 데이터를 Pandas DataFrame으로 변환
    data = pd.DataFrame({
        "x": x,  # x 값
        "y": y   # y 값
    })

    # 데이터를 CSV 파일로 저장
    csv_filename = "t_distribution_xy_data.csv"  # 저장할 파일 이름
    data.to_csv(csv_filename, index=False)  # 인덱스를 포함하지 않고 저장
    print(f"데이터가 {csv_filename} 파일로 저장되었습니다.")  # 저장 완료 메시지

    # 데이터 시각화
    plt.figure(figsize=(8, 6))  # 그래프 크기 설정
    plt.scatter(x, y, alpha=0.5, s=10, color="blue", label="t-분포 데이터")  # 산점도 그래프 그리기
    plt.title(f"t-분포 데이터 시각화 (자유도: {degrees_of_freedom})", fontsize=16)  # 그래프 제목
    plt.xlabel("x 값 (t-분포)", fontsize=14)  # X축 라벨
    plt.ylabel("y 값 (t-분포)", fontsize=14)  # Y축 라벨
    plt.legend(fontsize=12)  # 범례 추가
    plt.grid(True, linestyle="--", alpha=0.7)  # 격자 추가
    plt.show()  # 그래프 창 띄우기