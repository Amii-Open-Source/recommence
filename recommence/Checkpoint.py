import os
import shutil
import pickle
import logging
from typing import Any, Dict, Callable, TypeVar

from recommence.Config import CheckpointConfig

T = TypeVar('T')

logger = logging.getLogger('recommence')

class Checkpoint:
    def __init__(self, config: CheckpointConfig):
        self._c = config
        self._data: Dict[str, Any] = {}

        self._load_if_exists()

    def __getitem__(self, name: str) -> Any:
        return self._data[name]

    def __setitem__(self, name: str, value: Callable[[], T]) -> T:
        self._data[name] = value()
        return self._data[name]

    def register(self, name: str, builder: Callable[[], T]) -> T:
        if name in self._data:
            return self._data[name]

        self._data[name] = builder()
        return self._data[name]

    def save(self) -> None:
        os.makedirs(self._c.save_path, exist_ok=True)

        data_path = f'{self._c.save_path}/{self._c.data_file}'
        try:
            with open(data_path, 'wb') as f:
                pickle.dump(self._data, f)

        except Exception as e:
            if not self._c.no_fail:
                raise Exception("Could not save the checkpoint") from e

        logger.info(f'Saving checkpoint at: {data_path}')

    def remove(self) -> None:
        target_path = self._c.save_path
        if os.path.exists(target_path):
            shutil.rmtree(target_path)

        logger.info(f'Removing checkpoint at: {target_path}')


    def _load_if_exists(self) -> None:
        os.makedirs(self._c.save_path, exist_ok=True)
        data_path = f'{self._c.save_path}/{self._c.data_file}'
        try:
            if os.path.exists(data_path):
                logger.debug(f'Checkpoint data-file found at: {data_path}')

                try:
                    with open(data_path, 'rb') as f:
                        self._data = pickle.load(f)
                except Exception as e:
                    if not self._c.no_fail:
                        raise Exception("Could not load the checkpoint") from e
        except Exception as e:
            if not self._c.no_fail:
                raise Exception(f"Could not find the file at the path: {data_path}") from e

        logger.info(f'Loading from checkpoint at: {self._c.save_path}')
