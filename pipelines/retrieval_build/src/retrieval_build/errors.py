from __future__ import annotations


class RetrievalError(RuntimeError):
    """Base error with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class IndexBuildError(RetrievalError):
    pass


class IndexNotReadyError(RetrievalError):
    pass
