from typing import Optional, Any


class Debug:
    _instance: Optional["Debug"] = None
    _enabled = False

    def __new__(cls) -> "Debug":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def enable(self) -> None:
        '''
        sets the flag to true
        '''
        Debug._enabled = True

    def dprint(self, *args: Any, **kwargs: Any) -> None:
        '''
        Function that accepts text and variables to perform a print operation.
        '''
        if Debug._enabled:
            print(*args, **kwargs)
