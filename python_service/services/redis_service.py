import json
import time
from typing import Any
from redis.asyncio import Redis
import asyncio
from config import settings
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class RedisService:
    REDIS_HOST = settings.REDIS_HOST
    REDIS_PORT = settings.REDIS_PORT
    REDIS_DB = settings.REDIS_DB
    STREAM_NAME = settings.STREAM_NAME
    GROUP_NAME = settings.GROUP_NAME
    GROUP_NAME_RETRY = settings.GROUP_NAME_RETRY
    RETRY_STREAM_NAME = settings.RETRY_STREAM_NAME
    EXTERNAL_API_URL = settings.EXTERNAL_API_URL
    def __init__(self, r: Redis = None):
    
        self.redis = r
        
        self.init_redis()


    # @staticmethod
    def init_redis(self):
        if(self.redis is None):
            self.redis = Redis(
                host=self.REDIS_HOST,
                port=self.REDIS_PORT,
                db=self.REDIS_DB,
                decode_responses=True
            )

    async def ping(self) -> bool:
        try:
            res = await self.redis.ping()
            if(res):
                return True
            else:
                return False
        except Exception as e:
            raise Exception(f"Error ping redis: {e}")
        
  

    async def add_to_stream(self, message: dict, stream_name: str, max_retries: int = 2):
        for i in range(max_retries):
            try:
                res = await self.redis.xadd(
                    name=stream_name,
                    fields=message
                )
                logger.info(f"Message added to '{stream_name}': {res}")
                return res
            except Exception as e:
                logger.error(f"Error adding message to {stream_name}: {e}")
                raise e
        logger.error(f"Failed to add message to {stream_name} after {max_retries} retries")

    async def get_stream_group(self, stream_name: str, group_name: str, 
                               consumer_name: str,
                               max_retries: int = 2):
        """Get a message from the stream group

        Args:
            stream_name (str): _description_
            group_name (str): _description_
            consumer_name (str): _description_
            max_retries (int, optional): _description_. Defaults to 2.

        Returns:
            _type_: _description_
            [
                [stream_name1, [message_id1, message1 :metadata]],
            ]
        """
        for i in range(max_retries):
            try:
                res = await self.redis.xreadgroup(
                    groupname=group_name,
                    consumername=consumer_name,
                    streams={stream_name: '>'},
                    count=1,
                    block=0
                )
                return res
            except Exception as e:
                raise e
        logger.error(f"Failed to read stream {stream_name} after {max_retries} retries")
        return None

    async def create_group(self, stream_name: str, group_name: str, max_retries: int = 3):
        try:
            
            res = await self.redis.xgroup_create(
                name=stream_name,
                groupname=group_name,
                id='$'
            )
            logger.info(f"Group {group_name} created for stream {stream_name}: {res}")
            return res
        except Exception as e:
            raise e

    async def get_pending_messages(self, stream_name: str, group_name: str, max_retries: int = 3):
        """Get all pending messages that haven't been acknowledged"""
        for i in range(max_retries):
            try:
                res = await self.redis.xpending(
                    name=stream_name,
                    groupname=group_name
                )
                logger.info(f"Pending messages in {stream_name} for group {group_name}: {res}")
                return res
            except Exception as e:
                logger.error(f"Error getting pending messages: {e}")
                logger.info(f"Retrying in 1 seconds")
                await asyncio.sleep(1)
        logger.error(f"Failed to get pending messages after {max_retries} retries")
        return None

    async def claim_pending_message(self, stream_name: str, group_name: str, 
                                   consumer_name: str, min_idle_time: int = 5000,
                                   max_retries: int = 3):
        """Claim a pending message that has been idle for too long"""
        for i in range(max_retries):
            try:
                res = await self.redis.xclaim(
                    name=stream_name,
                    groupname=group_name,
                    consumername=consumer_name,
                    min_idle_time=min_idle_time,
                    message_ids=['0-0']  # This will be replaced with actual message ID
                )
                logger.info(f"Claimed pending message in {stream_name}: {res}")
                return res
            except Exception as e:
                logger.error(f"Error claiming pending message: {e}")
                logger.info(f"Retrying in 1 seconds")
                await asyncio.sleep(1)
        logger.error(f"Failed to claim pending message after {max_retries} retries")
        return None

    async def acknowledge_message(self, stream_name: str, group_name: str, 
                                message_id: str, max_retries: int = 3):
        """Acknowledge a message after successful processing"""
        for i in range(max_retries):
            try:
                res = await self.redis.xack(
                    stream_name,
                    group_name,
                    message_id
                )
                return res
            except Exception as e:
                logger.error(f"Error acknowledging message: {e}")
        logger.error(f"Failed to acknowledge message after {max_retries} retries")
        return None

    async def auto_claim_messages(self, stream_name: str, group_name: str,
                                consumer_name: str, min_idle_time: int = 1800000,
                                count: int = 100, max_retries: int = 3):
        """Automatically claim pending messages that have been idle for too long
        This uses XAUTOCLAIM which is available in Redis 6.2+
        """
        for i in range(max_retries):
            try:
                # XAUTOCLAIM returns a tuple of (next_id, messages)
                next_id, messages, deleted_ids = await self.redis.xautoclaim(
                    name=stream_name,
                    groupname=group_name,
                    consumername=consumer_name,
                    min_idle_time=min_idle_time,
                    count=count,
                    start_id='0-0'  # Start from the beginning of the stream
                )
                if(len(messages) > 0):
                    logger.info(f"Auto-claimed {len(messages)} messages in {stream_name}")
                    return next_id, messages
                else:
                    return None, []
            except Exception as e:
                logger.error(f"Error auto-claiming messages: {e}")
        logger.error(f"Failed to auto-claim messages after {max_retries} retries")
        return None, []

    async def get_pending_messages_with_time(self, stream_name: str, group_name: str, max_retries: int = 3):
        """Get pending messages with their idle time"""
        for i in range(max_retries):
            try:
                # XPENDING with -1 count returns all pending messages
                res = await self.redis.xpending(
                    name=stream_name,
                    groupname=group_name,
                    count=-1,
                    consumername=None,  # Get messages from all consumers
                    idle=0  # Get all messages regardless of idle time
                )
                logger.info(f"Got pending messages with time info: {res}")
                return res
            except Exception as e:
                logger.error(f"Error getting pending messages with time: {e}")
                logger.info(f"Retrying in 1 seconds")
                await asyncio.sleep(1)
        logger.error(f"Failed to get pending messages with time after {max_retries} retries")
        return None

    async def claim_specific_message(self, stream_name: str, group_name: str,
                                   consumer_name: str, message_id: str,
                                   max_retries: int = 3):
        """Claim a specific message that has been idle for too long"""
        for i in range(max_retries):
            try:
                res = await self.redis.xclaim(
                    name=stream_name,
                    groupname=group_name,
                    consumername=consumer_name,
                    min_idle_time=0,  # We don't care about idle time here
                    message_ids=[message_id]
                )
                logger.info(f"Claimed specific message {message_id} in {stream_name}: {res}")
                return res
            except Exception as e:
                logger.error(f"Error claiming specific message: {e}")
                logger.info(f"Retrying in 1 seconds")
                await asyncio.sleep(1)
        logger.error(f"Failed to claim specific message after {max_retries} retries")
        return None

    async def mark_message_processing(self, stream_name: str, message_id: str, 
                                    consumer_name: str, max_retries: int = 3):
        """Mark a message as being processed by adding metadata"""
        for i in range(max_retries):
            try:
                # Add processing metadata to the message
                processing_key = f"processing:{stream_name}:{message_id}"
                res = await self.redis.hset(
                    processing_key,
                    mapping={
                        "status": "processing",
                        "consumer": consumer_name,
                        "start_time": int(time.time() * 1000),  # Current time in milliseconds
                        "last_update": int(time.time() * 1000)
                    }
                )
                # Set expiration for the processing metadata (e.g., 1 hour)
                await self.redis.expire(processing_key, 3600)
                logger.info(f"Marked message {message_id} as processing: {res}")
                return res
            except Exception as e:
                logger.error(f"Error marking message as processing: {e}")
                logger.info(f"Retrying in 1 seconds")
                await asyncio.sleep(1)
        logger.error(f"Failed to mark message as processing after {max_retries} retries")
        return None

    async def update_message_processing(self, stream_name: str, message_id: str, 
                                      max_retries: int = 3):
        """Update the last_update time of a processing message"""
        for i in range(max_retries):
            try:
                processing_key = f"processing:{stream_name}:{message_id}"
                res = await self.redis.hset(
                    processing_key,
                    "last_update",
                    int(time.time() * 1000)
                )
                logger.info(f"Updated processing time for message {message_id}: {res}")
                return res
            except Exception as e:
                logger.error(f"Error updating message processing time: {e}")
                logger.info(f"Retrying in 1 seconds")
                await asyncio.sleep(1)
        logger.error(f"Failed to update message processing time after {max_retries} retries")
        return None

    async def mark_message_completed(self, stream_name: str, message_id: str, 
                                   max_retries: int = 3):
        """Mark a message as completed processing"""
        for i in range(max_retries):
            try:
                processing_key = f"processing:{stream_name}:{message_id}"
                res = await self.redis.hset(
                    processing_key,
                    mapping={
                        "status": "completed",
                        "end_time": int(time.time() * 1000)
                    }
                )
                logger.info(f"Marked message {message_id} as completed: {res}")
                return res
            except Exception as e:
                logger.error(f"Error marking message as completed: {e}")
                logger.info(f"Retrying in 1 seconds")
                await asyncio.sleep(1)
        logger.error(f"Failed to mark message as completed after {max_retries} retries")
        return None

    async def get_stuck_messages(self, stream_name: str, group_name: str,
                               max_idle_time: int = 300000,  # 5 minutes
                               max_processing_time: int = 1800000,  # 30 minutes
                               max_retries: int = 3):
        """Get messages that are likely stuck (either idle too long or processing too long)"""
        for i in range(max_retries):
            try:
                # Get all pending messages with their details
                pending_messages = await self.redis.xpending(
                    name=stream_name,
                    groupname=group_name,
                    count=-1,  # Get all messages
                    consumername=None,  # From all consumers
                    idle=0  # Regardless of idle time
                )

                stuck_messages = []
                current_time = int(time.time() * 1000)

                for message in pending_messages:
                    message_id, consumer, idle_time, delivery_count = message
                    
                    # Check processing metadata
                    processing_key = f"processing:{stream_name}:{message_id}"
                    processing_info = await self.redis.hgetall(processing_key)

                    if processing_info:
                        # Message has processing metadata
                        start_time = int(processing_info.get('start_time', 0))
                        last_update = int(processing_info.get('last_update', 0))
                        status = processing_info.get('status', '')

                        # Check if processing time exceeds limit
                        if status == 'processing' and (current_time - last_update) > max_processing_time:
                            stuck_messages.append({
                                'message_id': message_id,
                                'consumer': consumer,
                                'reason': 'processing_timeout',
                                'processing_time': current_time - start_time,
                                'last_update': current_time - last_update
                            })
                    else:
                        # No processing metadata, check idle time
                        if idle_time > max_idle_time:
                            stuck_messages.append({
                                'message_id': message_id,
                                'consumer': consumer,
                                'reason': 'idle_timeout',
                                'idle_time': idle_time
                            })

                logger.info(f"Found {len(stuck_messages)} stuck messages")
                return stuck_messages
            except Exception as e:
                logger.error(f"Error getting stuck messages: {e}")
                logger.info(f"Retrying in 1 seconds")
                await asyncio.sleep(1)
        logger.error(f"Failed to get stuck messages after {max_retries} retries")
        return None

     
