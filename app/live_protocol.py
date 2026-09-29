"""Length-delimited CDR transport. One acknowledged packet may be in flight."""

import json
import struct

MAX_CDR_BYTES = 32 * 1024 * 1024


def receive_exact(sock, count):
    data = bytearray()
    while len(data) < count:
        part = sock.recv(count - len(data))
        if not part:
            raise EOFError("ROS transport closed")
        data.extend(part)
    return bytes(data)


def send_json(sock, value):
    data = json.dumps(value, allow_nan=False).encode()
    sock.sendall(struct.pack("!I", len(data)) + data)


def receive_json(sock):
    size = struct.unpack("!I", receive_exact(sock, 4))[0]
    if size > 65536:
        raise ValueError("Transport metadata exceeds limit")
    return json.loads(receive_exact(sock, size))


def receive_cloud(sock):
    meta = receive_json(sock)
    size = meta.get("bytes", -1)
    if not isinstance(size, int) or not 0 < size <= MAX_CDR_BYTES:
        raise ValueError("Invalid CDR payload size")
    return meta, receive_exact(sock, size)
