import os
import logging
from recommence.Config import ReporterConfig


sample_reporter_config = ReporterConfig(
    types=['file'],
    file_save_path="tmp/report.txt",
    database_path=None,
    logger=logging.getLogger()
)

class Reporter:
    def __init__(self, config: ReporterConfig = sample_reporter_config):
        self.metrics: dict = {}
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
            with open(self.reporter_config.get_report_path(), 'w') as f:
                for stage, value in self.metrics.items():
                    f.write(f'{stage}: {value}\n')

        elif 'sql' in self.reporter_config.types:
            raise NotImplementedError("SQL reporting is not yet implemented")



    def report(self):
        if self.reporter_config.should_report:
            self._report()
        else:
            return




