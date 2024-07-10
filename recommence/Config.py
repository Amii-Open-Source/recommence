from dataclasses import dataclass

@dataclass
class CheckpointConfig:
    save_path: str
    staging_path: str | None = None
    data_file: str = 'data.pkl'

    no_fail: bool = False

    def should_stage(self) -> bool:
        return self.staging_path is not None

    def get_staging_path(self) -> str:
        if self.staging_path is not None:
            return self.staging_path

        return self.save_path

    def get_staging_data_path(self) -> str:
        return f'{self.get_staging_path()}/{self.data_file}'
