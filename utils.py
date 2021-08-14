import datetime


def parse_time(x):
    try:
        splited = [int(y) for y in x.split(":")]
        time_parsed = datetime.time(*splited)
        time_parsed = datetime.timedelta(
            hours=time_parsed.hour,
            minutes=time_parsed.minute,
            seconds=time_parsed.second,
        )
        return time_parsed.total_seconds()
    except ValueError:
        return -1
