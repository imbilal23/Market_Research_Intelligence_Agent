def provider_error_message(error: Exception, fallback: str) -> str:
    """Translate provider failures without exposing credentials or internal diagnostics."""
    message = str(error).lower()
    if "429" in message or "resource_exhausted" in message:
        if "per day" in message or "perday" in message:
            return "The AI service has reached its daily usage limit. Please try again after the quota resets, or ask the service owner to check the API quota."
        return "The AI service has reached its usage limit. Please try again later. If this continues, the service owner needs to check the API quota."
    if isinstance(error, TimeoutError) or any(word in message for word in ("timeout", "timed out", "deadline_exceeded")):
        return "The AI service took too long to respond. Please try again."
    if "503" in message or "unavailable" in message:
        return "The AI service is temporarily unavailable. Please try again shortly."
    return fallback
