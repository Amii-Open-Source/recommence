import os
import shutil
import pickle
from typing import Any, Dict, Callable, TypeVar

T = TypeVar('T')
class Checkpoint:
    def __init__(self, save_path: str):
        self.save_path: str = save_path
        self._data_path: str = f'{save_path}/data.pkl'
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
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        with open(self._data_path, 'wb') as f:
            pickle.dump(self._data, f)

    def remove(self) -> None:
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
        return

    def _load_if_exists(self) -> None:
        if os.path.exists(self.save_path):
            if os.path.exists(self._data_path):
                with open(self._data_path, 'rb') as f:
                    self._data = pickle.load(f)
        return
