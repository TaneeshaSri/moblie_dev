import os
import asyncio
from aiokafka.abc import AbstractTokenProvider
from aws_msk_iam_sasl_signer import MSKAuthTokenProvider

from app import config


def oauth_cb(oauth_config):
    auth_token, expiry_ms = MSKAuthTokenProvider.generate_auth_token(
        config.KAFKA_AWS_REGION
    )

    return auth_token, expiry_ms / 1000


class AWSTokenProvider(AbstractTokenProvider):
    async def token(self):
        return await asyncio.get_running_loop().run_in_executor(None, self._token)

    def _token(self):
        token, _ = MSKAuthTokenProvider.generate_auth_token(config.KAFKA_AWS_REGION)
        return token
