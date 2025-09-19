class ParallellmSignal(Exception):
    """
    Parallellm uses exceptions as "signals" to prevent
    code from executing.

    They should always be automatically caught as long as you are using
    the BatchManager inside a 'with' block.
    """


class WrongCheckpoint(ParallellmSignal):
    def __init__(self, message=None):
        """
        If you are seeing this exception, you are doing something wrong.

        All BatchManager code should be run inside a 'with' block,
        which should automatically catch these exceptions.
        """
        if message is None:
            super().__init__(
                "This exception should not be seen. "
                "You should be calling Parallellm.when_checkpoint() inside a 'with' block."
            )
        else:
            super().__init__(message)


class NotAvailable(ParallellmSignal):
    pass
