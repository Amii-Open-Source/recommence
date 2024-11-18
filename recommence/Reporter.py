import os
import logging
from typing import Any
from recommence.Config import ReporterConfig
from abc import abstractmethod
import sqlite3

sample_reporter_config = ReporterConfig(
    types=['file', 'logger', 'sql'],
    file_save_path="tmp/report.txt",
    database_path="tmp/report.db",
    logger=logging.getLogger()
)

class Reporter:
    def __init__(self, config: ReporterConfig = sample_reporter_config):
        self.metrics = {}
        self.reporter_config: ReporterConfig = config


    def report_size(self, stage: str, path: str):
        stage = stage.split(":")[1]
        self.metrics[stage] = os.path.getsize(path)
        return self.metrics[stage]


    def _report(self):
        for type in self.reporter_config.types:
            if type == 'logger':
                LoggerReporterBackend().write(self.reporter_config, self.metrics)
            elif type == 'file':
                FileReporterBackend().prep(self.reporter_config)
                FileReporterBackend().write(self.reporter_config, self.metrics)
            elif type == 'sql':
                SQLReporterBackend().prep(self.reporter_config)
                SQLReporterBackend().write(self.reporter_config, self.metrics)



    def report(self):
        if self.reporter_config.should_report:
            self._report()
        else:
            return


class ReporterBackend:
    @abstractmethod
    def prep(self, reporter_config) -> None:
        ...

    @abstractmethod
    def write(self, reporter_config, metrics) -> None:
        ...


class FileReporterBackend(ReporterBackend):
  def prep(self, reporter_config: ReporterConfig):
      if reporter_config.file_save_path is not None:
        os.makedirs(os.path.dirname(reporter_config.file_save_path), exist_ok = True)

  def write(self, reporter_config: ReporterConfig, metrics: dict[str, Any]):
      with open(reporter_config.get_report_path(), 'w') as f:
          for stage, value in metrics.items():
              f.write(f'{stage}: {value}\n')


class LoggerReporterBackend(ReporterBackend):
    def prep(self, reporter_config: ReporterConfig):
        pass

    def write(self, reporter_config: ReporterConfig, metrics: dict[str, Any]):
        logger = reporter_config.get_logger()
        for stage, value in metrics.items():
            logger.info(f'{stage}: {value}')


class SQLReporterBackend(ReporterBackend):
    _instance = None #make it singleton
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def prep(self, reporter_config: ReporterConfig):
        if reporter_config.database_path is None:
            return

        self.conn = sqlite3.connect(reporter_config.database_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("CREATE TABLE IF NOT EXISTS metrics (stage TEXT, value REAL)")

    def write(self, reporter_config: ReporterConfig, metrics: dict[str, Any]):
        if reporter_config is None:
            return

        for stage, value in metrics.items():
            self.cursor.execute("INSERT INTO metrics (stage, value) VALUES (?, ?)", (stage, value))
        self.conn.commit()
        self.conn.close()
