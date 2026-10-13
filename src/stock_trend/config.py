from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectConfig:
    data_dir: Path = PROJECT_ROOT / "data"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    lookback: int = 30
    horizon: int = 30
    validation_ratio: float = 0.2
    batch_size: int = 128
    epochs: int = 30
    steps_per_epoch: int = 2000
    validation_steps: int = 400
    threshold: float = 0.5

    @property
    def train_path(self) -> Path:
        return self.data_dir / "train.csv"

    @property
    def test_path(self) -> Path:
        return self.data_dir / "test.csv"

    @property
    def sample_submission_path(self) -> Path:
        return self.data_dir / "sample_submission.csv"

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / "best_model.keras"

    @property
    def submission_path(self) -> Path:
        return self.outputs_dir / "submission.csv"

    def create_directories(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
