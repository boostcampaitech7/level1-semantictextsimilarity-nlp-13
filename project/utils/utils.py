import os
from datetime import datetime

# 실험마다 새로운 폴더에 보관하기 위한 기능
## experiments 폴더 내부에 하위 폴더를 생성하고 path를 전달합니다
def create_experiment_folder(CFG, base_path="./experiments"):
    # 현재 시간 기록
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # 날짜와 시간만 표시
    date_and_hour = current_time[5:13]

    # admin 값을 가져와서 폴더 이름에 추가
    admin_name = CFG['admin']

    experiment_folder_name = f"{date_and_hour}_{admin_name}"

    # experiments 경로에 해당 폴더 생성
    experiment_path = os.path.join(base_path, experiment_folder_name)
    os.makedirs(experiment_path, exist_ok=True)
    
    return experiment_path