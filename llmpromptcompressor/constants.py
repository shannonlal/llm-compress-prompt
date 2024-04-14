from enum import Enum

class RankMethodType(Enum):
    OPEN_AI = "OPEN_AI"
    CLAUDE = "CLAUDE"
    BM25 = "BM25"