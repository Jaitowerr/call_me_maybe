class Debug:
    _instance = None
    _enabled = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def enable(self):
        Debug._enabled = True

    # def disable(self):
    #     Debug._enabled = False

    def dprint(self, *args, **kwargs):
        if Debug._enabled:
            print(*args, **kwargs)
