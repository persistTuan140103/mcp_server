import logging
import uuid
from redis.asyncio import Redis
from httpx import AsyncClient
from typing import Any
import json
from redis_stream.logger_config import setup_logging
from services.redis_service import RedisService

setup_logging()

logger = logging.getLogger(__name__)

GROUP_NAME = RedisService.GROUP_NAME
GROUP_NAME_RETRY = RedisService.GROUP_NAME_RETRY
STREAM_NAME = RedisService.STREAM_NAME
RETRY_STREAM_NAME = RedisService.RETRY_STREAM_NAME
CONSUMER_NAME = "consumer_main_" + str(uuid.uuid4())

r = Redis(
    host=RedisService.REDIS_HOST,
    port=RedisService.REDIS_PORT,
    db=RedisService.REDIS_DB,
    decode_responses=True
)

redis_service = RedisService(r)

client = AsyncClient(base_url=RedisService.EXTERNAL_API_URL)


async def process_message(documents: Any) -> bool:
    """
    Process message from Redis, with metadata is list document from /send-list-to-redis
    """
    try:
        if("page_content" not in documents or "metadata" not in documents):
            logger.error(f"message is not correct format, do not process")
            return False
        new_documents = [
            {
                "page_content": documents["page_content"],
                "metadata": json.loads(documents["metadata"])
            }
        ]
        embedding_response = await client.post("/embedd-and-save", json=new_documents)
        
        ## continue process save to vector database qdrant  
        if embedding_response.status_code == 200:
            return True
        else:
            return False
    except Exception as e:
        logger.error(f"Error embedding document: {e}")
        return False

async def init_redis():
    try:
        await redis_service.add_to_stream(
            message={"_init": "true"},
            stream_name=STREAM_NAME
        )
        await redis_service.create_group(
            stream_name=STREAM_NAME,
            group_name=GROUP_NAME
        )
    except Exception as e:
        logger.error(f"Error creating group {GROUP_NAME}: {e}")
    

async def main():
    logger.info(f"Starting consumer {CONSUMER_NAME}")
    while True:
        
        """
        with stream_name is stream_main then get list message with batch size
        """
        message = await redis_service.get_stream_group(
            stream_name=STREAM_NAME,
            group_name=GROUP_NAME,
            consumer_name=CONSUMER_NAME
        )
        if message:
            try:
                documents = message[0][1][0][1] #metadata is list document from /send-list-to-redis
                # print(documents)
                if await process_message(documents=documents):
                    if(await redis_service.acknowledge_message(
                        stream_name=STREAM_NAME,
                        group_name=GROUP_NAME,
                        message_id=message[0][1][0][0] # get message Id
                    )):
                        logger.info(f"message from '{STREAM_NAME}' is acknowledged: {message[0][1][0][0]}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await redis_service.add_to_stream(
                    stream_name=RETRY_STREAM_NAME,
                    group_name=GROUP_NAME_RETRY,
                    message={
                        **message,
                        "original_stream": STREAM_NAME,
                        "error": str(e)
                    }
                )
                
                await redis_service.acknowledge_message(
                    stream_name=STREAM_NAME,
                    group_name=GROUP_NAME,
                    message_id=message[0][1][0][0]
                )
        await asyncio.sleep(1)

async def run_all_task():
    try:
            
        await init_redis()
        await main()
    except Exception as e:
        logger.error(f"Error running all task: {e}")
        raise e


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_all_task())