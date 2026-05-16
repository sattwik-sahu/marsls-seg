from datetime import datetime as dt


def get_timestamp_now() -> str:
    return dt.now().strftime(format="%Y%m%d-%H%M%S")
