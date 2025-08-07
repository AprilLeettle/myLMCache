# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
    parse_remote_url,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)


class ParastorConnectorAdapter(ConnectorAdapter):
    """Adapter for Parastor connectors."""

    def __init__(self) -> None:
        super().__init__("parastor://")

    def can_parse(self, url: str) -> bool:
        return url.startswith(self.schema)

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .parastor_connector import ParastorConnector

        logger.info(f"Creating Parastor connector for URL: {context.url}")
        
        parse_url = parse_remote_url(context.url)
        return ParastorConnector(
            parse_url.path, 
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend
        )
