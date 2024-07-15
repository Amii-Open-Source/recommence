import os
import shutil
import pickle
import logging
from typing import Any, Dict, Callable, TypeVar

T = TypeVar('T')

logger = logging.getLogger('recommence')

class Checkpoint:
    def __init__(self, save_path: str, no_fail: bool = False) -> None:
        self.save_path: str = save_path
        self._data_path: str = f'{save_path}/data.pkl'
        self._data: Dict[str, Any] = {}

        self.no_fail: bool = no_fail  # If true, do not fail if there are issues
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
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        try:
            with open(self._data_path, 'wb') as f:
                pickle.dump(self._data, f)
        except Exception as e:
            if not self.no_fail:
                raise Exception("Could not save the checkpoint") from e

        logger.info(f'Saving checkpoint at: {self.save_path}')

    def remove(self) -> None:
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)

        logger.info(f'Removing checkpoint at: {self.save_path}')

    def _load_if_exists(self) -> None:
        if os.path.exists(self.save_path):
            logger.debug(f'Checkpoint directory found at: {self.save_path}')

            try:
                if os.path.exists(self._data_path):
                    logger.debug(f'Checkpoint data-file found at: {self._data_path}')

                    try:
                        with open(self._data_path, 'rb') as f:
                            self._data = pickle.load(f)
                    except Exception as e:
                        if not self.no_fail:
                            raise Exception("Could not load the checkpoint") from e
            except Exception as e:
                if not self.no_fail:
                    raise Exception(f"Could not find the file at the path: {self._data_path}") from e

        logger.info(f'Loading from checkpoint at: {self.save_path}')
