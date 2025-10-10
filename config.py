import os
from dotenv import load_dotenv
load_dotenv()
MODEL = os.getenv("MODEL")
EMBED_MODEL = os.getenv("EMBED_MODEL")
assert MODEL, "Please set MODEL in your .env to a supported model id"