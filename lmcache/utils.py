# SPDX-License-Identifier: Apache-2.0
# Future
from __future__ import annotations

import asyncio
from concurrent.futures import Future
from itertools import cycle
# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple
import hashlib
import threading

# Third Party
from nvtx import annotate  # type: ignore
import torch

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.memory_management import MemoryFormat

# Type definition
KVCache = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


@dataclass
class DiskCacheMetadata:
    path: str
    size: int  # in bytes
    shape: Optional[torch.Size] = None
    dtype: Optional[torch.dtype] = None
    fmt: MemoryFormat = None
    pin_count: int = 0

    def pin(self) -> bool:
        self.pin_count += 1
        return True

    def unpin(self) -> bool:
        self.pin_count -= 1
        return True

    @property
    def is_pinned(self) -> bool:
        return self.pin_count > 0


TORCH_DTYPE_TO_STR_DTYPE = {
    torch.half: "half",
    torch.float16: "half",
    torch.bfloat16: "bfloat16",
    torch.float: "float",
    torch.float32: "float",
    torch.float64: "double",
    torch.double: "double",
    torch.uint8: "fp8",
    torch.float8_e4m3fn: "fp8_e4m3",
    torch.float8_e5m2: "fp8_e5m2",
}

STR_DTYPE_TO_TORCH_DTYPE = {v: k for k, v in TORCH_DTYPE_TO_STR_DTYPE.items()}


@dataclass(order=True)
class CacheEngineKey:
    fmt: str
    model_name: str
    world_size: int
    worker_id: int
    chunk_hash: int

    def __hash__(self):
        return hash(
            (
                self.fmt,
                self.model_name,
                self.world_size,
                self.worker_id,
                self.chunk_hash,
            )
        )

    def to_string(self):
        return (
            f"{self.fmt}@{self.model_name}@{self.world_size}"
            f"@{self.worker_id}@{self.chunk_hash}"
        )

    def split_layers(self, num_layers: int) -> List["LayerCacheEngineKey"]:
        """Split the key into multiple keys for each layer"""
        keys = []
        for layer_id in range(num_layers):
            keys.append(
                LayerCacheEngineKey(
                    self.fmt,
                    self.model_name,
                    self.world_size,
                    self.worker_id,
                    self.chunk_hash,
                    layer_id,
                )
            )
        return keys

    def get_first_layer(self) -> "LayerCacheEngineKey":
        """Return the key for the first layer"""
        key = LayerCacheEngineKey(
            self.fmt,
            self.model_name,
            self.world_size,
            self.worker_id,
            self.chunk_hash,
            0,
        )
        return key

    @staticmethod
    def from_string(s):
        parts = s.split("@")
        if len(parts) != 5:
            raise ValueError(f"Invalid key string: {s}")
        return CacheEngineKey(
            parts[0], parts[1], int(parts[2]), int(parts[3]), int(parts[4])
        )

    def to_dict(self):
        # Note(Kuntai): this is used for serializing CacheEngineKey via msgpack.
        return {
            "__type__": "CacheEngineKey",
            "fmt": self.fmt,
            "model_name": self.model_name,
            "world_size": self.world_size,
            "worker_id": self.worker_id,
            "chunk_hash": self.chunk_hash,
        }

    @staticmethod
    def from_dict(d):
        return CacheEngineKey(
            fmt=d["fmt"],
            model_name=d["model_name"],
            world_size=d["world_size"],
            worker_id=d["worker_id"],
            chunk_hash=d["chunk_hash"],
        )


@dataclass(order=True)
class LayerCacheEngineKey(CacheEngineKey):
    """A key for the layer cache engine"""

    layer_id: int

    def __hash__(self):
        return hash(
            (
                self.fmt,
                self.model_name,
                self.world_size,
                self.worker_id,
                self.chunk_hash,
                self.layer_id,
            )
        )

    def to_string(self):
        return (
            f"{self.fmt}@{self.model_name}@{self.world_size}"
            f"@{self.worker_id}@{self.chunk_hash}@{self.layer_id}"
        )

    def split_layers(self, num_layers: int) -> List["LayerCacheEngineKey"]:
        """Split the key into multiple keys for each layer"""
        keys = []
        for layer_id in range(num_layers):
            keys.append(
                LayerCacheEngineKey(
                    self.fmt,
                    self.model_name,
                    self.world_size,
                    self.worker_id,
                    self.chunk_hash,
                    layer_id,
                )
            )
        return keys

    @staticmethod
    def from_string(s):
        parts = s.split("@")
        if len(parts) != 6:
            raise ValueError(f"Invalid key string: {s}")
        return LayerCacheEngineKey(
            parts[0],
            parts[1],
            int(parts[2]),
            int(parts[3]),
            int(parts[4]),
            int(parts[5]),
        )


##### NVTX annotation #####
_NVTX_COLORS = ["green", "blue", "purple", "rapids"]


def _get_color_for_nvtx(name):
    m = hashlib.sha256()
    m.update(name.encode())
    hash_value = int(m.hexdigest(), 16)
    idx = hash_value % len(_NVTX_COLORS)
    return _NVTX_COLORS[idx]


def _lmcache_nvtx_annotate(func, domain="lmcache"):
    """Decorator for applying nvtx annotations to methods in lmcache."""
    return annotate(
        message=func.__qualname__,
        color=_get_color_for_nvtx(func.__qualname__),
        domain=domain,
    )(func)


##### Observability Threading related #####
_shared_observability_lock = threading.Lock()


def thread_safe(func):
    def wrapper(*args, **kwargs):
        with _shared_observability_lock:
            result = func(*args, **kwargs)
        return result

    return wrapper

class RoundRobinEventLoopPool:
    def __init__(self, num_loops: int):
        self.loops: Dict[int, asyncio.AbstractEventLoop] = {}
        self.threads: Dict[int, threading.Thread] = {}
        self._loop_ids = list(range(num_loops))
        self._rr_cycle = cycle(self._loop_ids)  # 轮询迭代器
        self._lock = threading.Lock()  # 保证轮询迭代器线程安全

        # 初始化事件循环线程
        for i in self._loop_ids:
            self._start_loop_thread(thread_id=i)

    def _start_loop_thread(self, thread_id: int):
        """启动单个事件循环线程"""
        loop = asyncio.new_event_loop()
        
        def run_loop():
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(
            target=run_loop,
            name=f"AioThread-{thread_id}",
            daemon=True
        )
        thread.start()
        
        self.loops[thread_id] = loop
        self.threads[thread_id] = thread

    def _get_next_loop_id(self) -> int:
        """线程安全地获取下一个轮询目标ID"""
        with self._lock:
            return next(self._rr_cycle)

    def submit_coroutine(
        self, 
        coro, 
        thread_id: Optional[int] = None
    ) -> Future:
        """
        提交协程到事件循环线程
        :param thread_id: 如果为None则自动轮询选择
        """
        target_id = thread_id if thread_id is not None else self._get_next_loop_id()
        loop = self.loops[target_id]
        return asyncio.run_coroutine_threadsafe(coro, loop)

    def stop(self):
        """安全关闭所有事件循环"""
        for loop in self.loops.values():
            loop.call_soon_threadsafe(loop.stop)
        for thread in self.threads.values():
            thread.join(timeout=1)
