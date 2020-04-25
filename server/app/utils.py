
def bad_request(message):
    return message, 400, None


def server_error(message):
    return message, 500, None
