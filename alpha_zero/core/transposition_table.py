from typing import Any, Tuple
from enum import Enum

class NodeType(Enum):
    """Enumeration for the type of node in the transposition table."""
    EXACT = 0
    LOWERBOUND = 1
    UPPERBOUND = 2

class TranspositionTable:
    """
    Transposition Table for storing and retrieving previously computed states.
    """
    def __init__(self, size=1000000):
        """
        Initialize the transposition table with a given size.

        Args:
            size: The maximum number of entries the table can hold.
        """
        self.table = {}
        self.size = size

    def store(self, zobrist_hash: Any, depth: int, value: float, flag: NodeType) -> None:
        """
        Store a new entry in the transposition table.

        Args:
            zobrist_hash: The hash of the current board state.
            depth: The depth at which this state was evaluated.
            value: The evaluation value of the current state.
            flag: The type of node (EXACT, LOWERBOUND, UPPERBOUND).
        """
        if len(self.table) >= self.size:
            self.table.pop(next(iter(self.table)))
        self.table[zobrist_hash] = (depth, value, flag)
    
    def lookup(self, zobrist_hash: Any) -> Tuple[int, float, NodeType]:
        """
        Retrieve an entry from the transposition table if it exists.

        Args:
            zobrist_hash: The hash of the current board state.

        Returns:
            A tuple containing the depth, value, and flag of the state if found,
            otherwise None.
        """
        return self.table.get(zobrist_hash)