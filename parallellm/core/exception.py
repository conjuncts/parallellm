class ParallellmSignal(Exception):
    """
    Parallellm uses exceptions as "signals" to prevent
    code from executing.

    They should always be automatically caught as long as you are using
    the BatchManager inside a 'with' block.
    """


class NotAvailable(ParallellmSignal):
    pass


class IntegrityError(Exception):
    """
    If you are seeing this exception, something changed between runs
    """

    pass
