import shutil
from pathlib import Path
from ultralytics import YOLO

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.absolute()

# 모델 가중치 경로
WEIGHTS_DIR = PROJECT_ROOT / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

# 프리트레인 모델 가중치 (없으면 자동 다운로드)
PRETRAINED_WEIGHTS = WEIGHTS_DIR / "yolov8l.pt"
if not PRETRAINED_WEIGHTS.exists():
    print(f"프리트레인 모델을 다운로드합니다: {PRETRAINED_WEIGHTS}")
    # ultralytics가 자동으로 다운로드하므로 경로만 지정
    model = YOLO("yolov8l.pt")  # 자동 다운로드
    # 다운로드된 파일을 weights 디렉토리로 복사 (선택사항)
else:
    model = YOLO(str(PRETRAINED_WEIGHTS))

# 데이터셋 설정 파일
DATA_YAML = PROJECT_ROOT / "data.yaml"

if not DATA_YAML.exists():
    print(f"경고: {DATA_YAML} 파일이 없습니다. data.yaml 템플릿을 참고하여 생성하세요.")
    exit(1)

# 학습 시작
print("\n" + "="*60)
print("🚀 YOLO v8 파인튜닝 시작")
print("="*60)
print(f"📦 프리트레인 모델: {PRETRAINED_WEIGHTS}")
print(f"📁 데이터셋 설정: {DATA_YAML}")
print(f"📊 학습 이미지: train/images")
print(f"📊 검증 이미지: valid/images")
print("="*60 + "\n")

results = model.train(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=1280,  # CPU 학습 시 640 권장 (1280은 메모리 부족 가능)
    batch=4,  # CPU 학습 시 적절한 배치 크기
    device='cpu',   # GPU 사용 시 '0' 또는 0, CPU면 'cpu' (CUDA 확인 후 수정 가능)
    workers=0,  # CPU 학습 시 workers=0 권장 (메모리 절약)
    plots=True,  # 학습 결과 그래프 생성
    save=True,
    save_period=10,  # 10 에포크마다 체크포인트 저장
    amp=False,  # CPU에서는 AMP 비활성화
)

# 학습 완료 후 최고 모델을 weights 디렉토리로 복사
if results and hasattr(results, 'save_dir'):
    best_model = Path(results.save_dir) / "weights" / "best.pt"
    if best_model.exists():
        dest_model = WEIGHTS_DIR / "best.pt"
        shutil.copy2(best_model, dest_model)
        print(f"\n✅ 학습된 모델이 복사되었습니다: {dest_model}")
        print(f"   원본 위치: {best_model}")
    else:
        print(f"\n⚠️  최고 모델을 찾을 수 없습니다: {best_model}")
else:
    # results가 없거나 save_dir이 없는 경우 기본 경로 확인
    default_best = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
    if default_best.exists():
        dest_model = WEIGHTS_DIR / "best.pt"
        shutil.copy2(default_best, dest_model)
        print(f"\n✅ 학습된 모델이 복사되었습니다: {dest_model}")
    else:
        print(f"\n⚠️  학습된 모델을 찾을 수 없습니다. 수동으로 확인하세요:")
        print(f"   예상 위치: {default_best}")