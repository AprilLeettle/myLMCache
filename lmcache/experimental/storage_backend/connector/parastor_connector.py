# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
from typing import List, TYPE_CHECKING, Optional, no_type_check
from dataclasses import dataclass
import aiofiles
import torch
import json
import operator
from functools import reduce

from lmcache.experimental.memory_management import (MemoryObj, BytesBufferMemoryObj)
from lmcache.experimental.storage_backend.abstract_backend import \
    StorageBackendInterface
from lmcache.experimental.storage_backend.local_cpu_backend import \
    LocalCPUBackend
from lmcache.logging import init_logger
from lmcache.utils import (CacheEngineKey, DiskCacheMetadata,
                           _lmcache_nvtx_annotate, RoundRobinEventLoopPool)

from lmcache.experimental.protocol import RemoteMetadata
from lmcache.experimental.storage_backend.connector.base_connector import \
    RemoteConnector

METADATA_BYTES_LEN = 28

if TYPE_CHECKING:
    from lmcache.experimental.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)

@dataclass
class ParaStorConfig:
    local_hostname: str
    global_segment_size: int
    local_buffer_size: int
    parastor_path: str

    @staticmethod
    def from_file(file_path: str) -> 'ParaStorConfig':
        """Load the config from a JSON file."""
        with open(file_path) as fin:
            config = json.load(fin)
        return ParaStorConfig(
            local_hostname=config.get("local_hostname"),
            global_segment_size=config.get("global_segment_size", 3355443200),
            local_buffer_size=config.get("local_buffer_size", 1073741824),
            parastor_path=config.get("parastor_path"),
        )

    @staticmethod
    def load_from_env() -> 'ParaStorConfig':
        """Load config from a file specified in the environment variable."""
        config_file_path = os.getenv('PARASTOR_CONFIG_PATH')
        if config_file_path is None:
            raise ValueError(
                "The environment variable 'PARASTOR_CONFIG_PATH' is not set.")
        return ParaStorConfig.from_file(config_file_path)

class ParastorConnector(RemoteConnector):

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        aiopool: RoundRobinEventLoopPool,
        local_cpu_backend: LocalCPUBackend,
    ):
        try:
            # 尝试查看ParaStor是否安装，mount | grep ParaStor
            import json
        except Exception as e:
            logger.error(f"ParaStor is not installed: {e}")
            raise e
        
        try:
            self.config = ParaStorConfig.load_from_env()
            logger.info(f"ParaStor config: {self.config}")
        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error(
                "An error occurred while loading the configuration: %s", exc)
            raise

        assert self.config.parastor_path is not None
        self.path: str = self.config.parastor_path
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            logger.info(f"Created Parastor disk cache directory: {self.path}")

        self.local_cpu_backend = local_cpu_backend
        self.loop = loop
        self.aiopool = aiopool
        self.usage = 0

    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> str:
        return self.path + key.to_string().replace("/", "-") + ".pt"

    async def exists(self, key: CacheEngineKey) -> bool:
        return os.path.exists(self._key_to_path(key))

    def remove(
        self,
        key: CacheEngineKey,
    ) -> None:
        path = self.dict[key].path

        size = os.path.getsize(path)
        self.usage -= size
        # self.stats_monitor.update_local_storage_usage(self.usage)
        os.remove(path)

    async def put(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ):
        """
        Convert KV to bytes and async store bytes to disk.
        """
        path = self._key_to_path(key)

        # Please use a function like `memory_obj.to_meta()`.
        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        metadata_bytes = RemoteMetadata(len(kv_bytes), kv_shape, kv_dtype,
                                        memory_format).serialize()
        assert len(metadata_bytes) == METADATA_BYTES_LEN

        size = len(kv_bytes) + METADATA_BYTES_LEN
        self.usage += size
        # self.stats_monitor.update_local_storage_usage(self.usage)
        logger.debug(f"parastor put kv path: {path}, len: {size}")

        try:
            async with aiofiles.open(path, 'ab') as f:
                await f.write(metadata_bytes)
                await f.write(kv_bytes)
        except Exception as e:
            logger.error(f"Failed to put key"
                         f"meta type: {type(metadata_bytes)},"
                         f"data: {type(kv_bytes)}: {e}")

        memory_obj.ref_count_down()

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        Load bytearray from disk.
        """
        path = self._key_to_path(key)
        logger.debug(f"parastor get from path:{path}")
        try:
            async with aiofiles.open(path, 'rb') as f:
                content = await f.read()
        except Exception as e:
            logger.error(f"parastor Failed to get kv from path:{path}. {e}")

        if content is None:
            logger.error(f"file exists but empty")
            return None

        retrieved_view = memoryview(content)
        metadata_bytes = retrieved_view[:METADATA_BYTES_LEN]
        if metadata_bytes is None or len(metadata_bytes) != METADATA_BYTES_LEN:
            logger.error(f"file exists but metadata failed")
            return None

        metadata = RemoteMetadata.deserialize(metadata_bytes)

       # memory_obj = self.local_cpu_backend.allocate(
       #     metadata.shape,
       #     metadata.dtype,
       #     metadata.fmt,
       # )
        memory_obj = BytesBufferMemoryObj(retrieved_view[METADATA_BYTES_LEN:], metadata)
        assert len(retrieved_view) == metadata.length + METADATA_BYTES_LEN
        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None
        logger.debug(f"parastor get kv path: {path}, len: {metadata.length}")
        if memory_obj.tensor is not None:
            assert metadata.dtype is not None
            num_elements = reduce(operator.mul, metadata.shape)
            temp_tensor = torch.frombuffer(content,
                                           dtype=metadata.dtype,
                                           offset=METADATA_BYTES_LEN,
                                           count=num_elements).reshape(
                                               metadata.shape)

            memory_obj.tensor.copy_(temp_tensor)
        return memory_obj

    @no_type_check
    async def list(self) -> List[str]:
        pass
    def close(self) -> None:
        return None
