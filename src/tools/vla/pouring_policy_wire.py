"""
TCP 帧协议：Isaac 仿真端 <-> OpenVLA+LoRA 策略端（分进程）。

帧格式（大端 uint32 长度，避免粘包）：
  握手：客户端发 HELLO_CLNT (8B)，服务端回 HELLO_SRV (8B)
  观测：服务端发 OIMG (4B) + u32 jpeg_len + jpeg_bytes
  动作：客户端发 ACT7 (4B) + u32 28 + 7×float32 小端
"""

from __future__ import annotations

import socket
import struct

import numpy as np

HELLO_CLNT = b"POURCLN1"
HELLO_SRV = b"POURSRV1"
TAG_OIMG = b"OIMG"
TAG_SKIP = b"SKIP"
TAG_ACT7 = b"ACT7"


def recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(f"对端关闭或 EOF（尚需 {n - len(buf)} 字节）")
        buf.extend(chunk)
    return bytes(buf)


def send_obs_jpeg(conn: socket.socket, jpeg_bytes: bytes) -> None:
    conn.sendall(TAG_OIMG + struct.pack(">I", len(jpeg_bytes)) + jpeg_bytes)


def send_skip(conn: socket.socket) -> None:
    """本仿真步不做 VLA 推理（或 JPEG 失败）；客户端仍须回传一帧 ACT7。

    PouringEnv 为相对笛卡尔增量：SKIP 时应对前 6 维发 0，夹爪可保持上一拍命令，避免同一增量被多步累加。"""
    conn.sendall(TAG_SKIP + struct.pack(">I", 0))


def recv_obs_jpeg_or_skip(conn: socket.socket) -> bytes | None:
    """返回 JPEG bytes；若为 SKIP 则返回 None。"""
    tag = recv_exact(conn, 4)
    (ln,) = struct.unpack(">I", recv_exact(conn, 4))
    if tag == TAG_SKIP:
        if ln != 0:
            raise ValueError(f"SKIP 帧长度应为 0，收到 {ln}")
        return None
    if tag != TAG_OIMG:
        raise ValueError(f"期望 OIMG/SKIP，收到 {tag!r}")
    return recv_exact(conn, ln)


def recv_obs_jpeg(conn: socket.socket) -> bytes:
    out = recv_obs_jpeg_or_skip(conn)
    if out is None:
        raise ValueError("收到 SKIP，期望 OIMG")
    return out


def send_action7(conn: socket.socket, action: bytes) -> None:
    if len(action) != 28:
        raise ValueError(f"action 须为 28 字节 (7×float32)，实际 {len(action)}")
    conn.sendall(TAG_ACT7 + struct.pack(">I", 28) + action)


def recv_action7(conn: socket.socket) -> bytes:
    tag = recv_exact(conn, 4)
    if tag != TAG_ACT7:
        raise ValueError(f"期望 ACT7，收到 {tag!r}")
    (ln,) = struct.unpack(">I", recv_exact(conn, 4))
    if ln != 28:
        raise ValueError(f"动作载荷应为 28，收到 {ln}")
    return recv_exact(conn, 28)


def handshake_server(conn: socket.socket) -> None:
    h = recv_exact(conn, len(HELLO_CLNT))
    if h != HELLO_CLNT:
        raise ValueError(f"握手失败：期望 {HELLO_CLNT!r}，收到 {h!r}")
    conn.sendall(HELLO_SRV)


def handshake_client(conn: socket.socket) -> None:
    conn.sendall(HELLO_CLNT)
    h = recv_exact(conn, len(HELLO_SRV))
    if h != HELLO_SRV:
        raise ValueError(f"握手失败：期望 {HELLO_SRV!r}，收到 {h!r}")
