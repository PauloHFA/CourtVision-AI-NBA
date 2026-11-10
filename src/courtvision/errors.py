from __future__ import annotations


class CourtVisionError(Exception):
    code = "error"
    status_code = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class SameTeamError(CourtVisionError):
    code = "same_team"
    status_code = 400


class NotFoundError(CourtVisionError):
    code = "not_found"
    status_code = 404


class ContestAlreadyFinalError(CourtVisionError):
    code = "contest_already_final"
    status_code = 409


class InvalidAsOfError(CourtVisionError):
    code = "invalid_as_of"
    status_code = 400


class ModelUnavailableError(CourtVisionError):
    code = "model_unavailable"
    status_code = 503


class PermanentAuthError(CourtVisionError):
    code = "source_auth"
    status_code = 503


class PermanentRequestError(CourtVisionError):
    code = "source_request"
    status_code = 502


class TemporaryApiError(CourtVisionError):
    code = "source_unavailable"
    status_code = 503
