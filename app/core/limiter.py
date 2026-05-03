from slowapi import Limiter
from slowapi.util import get_remote_address

# Limit init
limiter = Limiter(key_func=get_remote_address)