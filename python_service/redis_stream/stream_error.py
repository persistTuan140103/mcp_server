import asyncio
import json
import uuid
from redis.asyncio import Redis
from typing import Any
from httpx import AsyncClient
import logging
from services.redis_service import RedisService
from redis_stream.logger_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

GROUP_NAME_RETRY = RedisService.GROUP_NAME_RETRY
# GROUP_NAME_MAIN = settings.GROUP_NAME
STREAM_NAME = RedisService.STREAM_NAME
RETRY_STREAM_NAME = RedisService.RETRY_STREAM_NAME
CONSUMER_NAME = "consumer_error_" + str(uuid.uuid4())

r = Redis(
    host=RedisService.REDIS_HOST,
    port=RedisService.REDIS_PORT,
    db=RedisService.REDIS_DB
)

http_client = AsyncClient(
    base_url=RedisService.EXTERNAL_API_URL
)

redis_service = RedisService(r)

async def process_message(documents: Any) -> bool:
    # Parse JSON string to dict
    if isinstance(documents, str):
        documents = json.loads(documents)
        
    if("page_content" not in documents or "metadata" not in documents):
        logger.error(f"message is not correct format, do not process")
        return False
    new_documents = [
        {
            "page_content": documents["page_content"],
            "metadata": documents["metadata"]
        }
    ]
    try:
        response = await http_client.post("/embedd-and-save", 
                         json=new_documents)
        if response.status_code == 200:
            return True
        else:
            return False
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return False
        

async def init_redis():
    try:
        await redis_service.add_to_stream(
            message={"_init": "true"},
            stream_name=RETRY_STREAM_NAME
        )
        await redis_service.create_group(
            stream_name=RETRY_STREAM_NAME,
            group_name=GROUP_NAME_RETRY
        )
    except Exception as e:
        logger.error(f"Error creating group {GROUP_NAME_RETRY}: {e}")
        

async def main():
    logger.info(f"Starting consumer {CONSUMER_NAME}")
    while True:
        try:
            message = await redis_service.get_stream_group(
                stream_name=RETRY_STREAM_NAME,
                group_name=GROUP_NAME_RETRY,
                consumer_name=CONSUMER_NAME
            )
            
            if(message is None):
                continue
            # print(type(message))
            print(message)
            # print("document: ", message[0][1][0][1])
            documents = message[0][1][0][1]
            if("documents" not in documents):
                logger.error(f"message is not correct format, do not process")
                continue
            if(await process_message(documents=documents["documents"])):
                if(await redis_service.acknowledge_message(
                    stream_name=RETRY_STREAM_NAME,
                    group_name=GROUP_NAME_RETRY,
                    message_id=message[0][1][0][0]
                )):
                    logger.info(f"message from '{RETRY_STREAM_NAME}' is acknowledged: {message[0][1][0][0]}")
            next_id, message_claim = await redis_service.auto_claim_messages(
                group_name=GROUP_NAME_RETRY,
                stream_name=RETRY_STREAM_NAME,
                consumer_name=CONSUMER_NAME,
                min_idle_time=600000, #10 minutes
                count=1,
                max_retries=1
            )
            
            if(message_claim is not None):
                if(await process_message(documents=documents)):
                    if(await redis_service.acknowledge_message(
                        stream_name=RETRY_STREAM_NAME,
                        group_name=GROUP_NAME_RETRY,
                        message_id=message_claim[0][1][0]
                        )):
                        logger.info(f"message from '{RETRY_STREAM_NAME}' is acknowledged: {message_claim[0][1][0]}")
            
            
        except Exception as e:
            logger.error(f"Error in main: {e}")
            continue
        await asyncio.sleep(5)

async def run_all_task():
    await init_redis()
    await main()

if __name__ == "__main__":
    asyncio.run(run_all_task())
    

