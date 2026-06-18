import json 
import os
import redis

STATE_KEY = "state"

def _get_redis_client() -> redis.Redis:
    """Get a Redis client connection with configuration from environment variables."""
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", 6379))
    password = os.getenv("REDIS_PASSWORD", None)
    db = int(os.getenv("REDIS_DB", 0))
    
    return redis.Redis(
        host=host,
        port=port,
        password=password,
        db=db,
        decode_responses=True
    )


def get_state() -> dict:
    """return the json.loads(value of STATE_KEY in redis)"""
    client = _get_redis_client()
    value = client.get(STATE_KEY)
    
    if value is None:
        return {}
    
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def set_state(state: dict) -> None:
    """set the value of STATE_KEY in redis to the state"""
    client = _get_redis_client()
    json_value = json.dumps(state)
    client.set(STATE_KEY, json_value)


def test():
    state = get_state()
    print(json.dumps(state, indent=4, ensure_ascii=False))
    
if __name__ == "__main__":
    test()