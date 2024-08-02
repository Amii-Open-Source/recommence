from dataclasses import dataclass
import logging
from typing import Any, List
@dataclass
class CheckpointConfig:
    save_path: str
    staging_path: str | None = None
    data_file: str = 'data.pkl'
    save_every_interval: int | None = None

    no_fail: bool = False

    def should_stage(self) -> bool:
        return self.staging_path is not None

    def get_staging_path(self) -> str:
        if self.staging_path is not None:
            return self.staging_path

        return self.save_path

    def get_staging_data_path(self) -> str:
        return f'{self.get_staging_path()}/{self.data_file}'

@dataclass
class ReporterConfig:
    '''
    config type is a list containing one or more of the following values:
    no_report: No report
    logger:  Through the logger
    file: To user specified file
    sql: To user specified sql database location
    '''
    types: List[str]
    logger: logging.Logger
    file_save_path: str | None = None
    database_path: str | None = None

    def should_report(self) -> bool:
        return "no_report" not in self.types

    def get_report_path(self) -> Any:
        if 'file' in self.types and self.file_save_path is not None:
            return self.file_save_path
        elif 'sql' in self.types and self.database_path is not None:
            return self.database_path
        elif 'logger' in self.types and self.logger is not None:
            return self.logger

    def get_logger(self) -> logging.Logger:
        return self.logger


