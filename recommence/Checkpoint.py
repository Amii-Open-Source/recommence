import os
import shutil
import pickle
import logging
import time
import signal
from typing import Any, Dict, Callable, TypeVar

from recommence.Config import CheckpointConfig
from recommence._utils.compress import compress_dir, uncompress_dir
from recommence._utils.pickle import read_pickle
from recommence.Reporter import Reporter

T = TypeVar('T')

logger = logging.getLogger('recommence')

class Checkpoint:
    def __init__(self, config: CheckpointConfig):
        self._c = config
        self._data: Dict[str, Any] = {}
        self._reporter = Reporter()
        self.external_data_paths = []
        self._last_save: float | None = None

        data = self._load_if_exists()
        if data is not None:
            self._data = data


    def __getitem__(self, name: str) -> Any:
        return self._data[name]

    def __setitem__(self, name: str, value: T | Callable[[], T]) -> T:
        if callable(value):
            self._data[name] = value()
        else:
            self._data[name] = value
        return self._data[name]

    def register(self, name: str, builder: Callable[[], T]) -> T:
        if name in self._data:
            return self._data[name]

        self._data[name] = builder()
        return self._data[name]

    def register_value(self, name: str, initial_value: T) -> T:
        if name in self._data:
            return self._data[name]

        self._data[name] = initial_value
        return self._data[name]

    def register_file(self, path: str) -> None:
        self.external_data_paths.append(path)

    def save(self) -> None:
        os.makedirs(self._c.get_staging_path(), exist_ok=True)

        data_path = self._c.get_staging_data_path()
        try:
            with open(data_path, 'wb') as f:
                pickle.dump(self._data, f)
                for path in self.external_data_paths:
                    shutil.copy(path, data_path)


        except Exception as e:
            if not self._c.no_fail:
                raise Exception("Could not save the checkpoint") from e

        logger.info(f'Saving checkpoint at: {data_path}')

        self._reporter.report_size("stage:size",data_path)
        self._reporter.report()


        # if only a save_path is given, then don't move the data
        if not self._c.should_stage():
            return

        compress_dir(
            input=self._c.get_staging_path(),
            target=self._c.save_path,
        )

    def maybe_save(self):
        if self._c.save_every_interval is None or self._c.save_every_interval < 0:
            return
        if self._last_save is None:
            self._last_save = time.time()

        if time.time() - self._last_save > self._c.save_every_interval:
            print("saving at maybe_save")
            self.save()
            self._last_save = time.time()

    def handle_preemption(self) -> None:
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum: int, frame: Any) -> None:
        logger.info('Received SIGTERM, saving checkpoint and exiting')
        try:
            self.save()
        except Exception as e:
            logger.error(f'Failed to save checkpoint: {e}')
        logger.info('Checkpoint saved gracefully, exiting')
        exit()



    def remove(self) -> None:
        # remove checkpoint from both target path
        # and staging path
        target_path = self._c.save_path
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
            logger.info(f'Removing checkpoint at: {target_path}')

        stage_path = self._c.get_staging_path()
        if os.path.exists(stage_path):
            shutil.rmtree(stage_path)
            logger.info(f'Removing checkpoint at: {stage_path}')

    def _load_if_exists(self) -> Dict[str, Any] | None:
        # if the checkpoint exists and is uncompressed, just load it
        data_path = f'{self._c.save_path}/{self._c.data_file}'
        if os.path.exists(data_path):
            logger.debug(f'Checkpoint data-file found at: {data_path}')
            for path in self.external_data_paths:
                shutil.copy(self._c.get_staging_path() + '/' + os.path.basename(path), path)
            return read_pickle(data_path, self._c.no_fail)

        # if there is a checkpoint path, but it is compressed
        if os.path.exists(f'{self._c.save_path}.tar.xz'):
            logger.info(f'Loading from checkpoint at: {self._c.save_path}.tar.xz')
            uncompress_dir(
                input=self._c.save_path,
                target=self._c.get_staging_path(),
            )

            for path in self.external_data_paths:
                shutil.copy(self._c.get_staging_path() + '/' + os.path.basename(path), path)

            return read_pickle(self._c.get_staging_data_path(), self._c.no_fail)
