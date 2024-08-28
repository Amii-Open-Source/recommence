import os
import logging
from recommence.Config import ReporterConfig
from abc import abstractmethod

sample_reporter_config = ReporterConfig(
    types=['file'],
    file_save_path="tmp/report.txt",
    database_path=None,
    logger=logging.getLogger()
)

class Reporter:
    def __init__(self, config = sample_reporter_config):
        self.metrics = {}
        self.reporter_config: ReporterConfig = config


    def report_size(self, stage, path):
        stage  = stage.split(":")[1]
        self.metrics[stage] = os.path.getsize(path)
        return self.metrics[stage]


    def _report(self):
        if 'logger' in self.reporter_config.types:
            logger = self.reporter_config.get_logger()
            for stage, value in self.metrics.items():
                logger.info(f'{stage}: {value}')

        elif 'file' in self.reporter_config.types:
            FileReporterBackend().prep(self.reporter_config)
            FileReporterBackend().write(self.reporter_config, self.metrics)

        elif 'sql' in self.reporter_config.types:
            raise NotImplementedError("SQL reporting is not yet implemented")



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
  def prep(self, reporter_config):
      os.makedirs(os.path.dirname(reporter_config.file_save_path), exist_ok = True)

  def write(self, reporter_config, metrics):
      with open(reporter_config.get_report_path(), 'w') as f:
          for stage, value in metrics.items():
              f.write(f'{stage}: {value}\n')
