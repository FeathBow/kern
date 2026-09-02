#!/usr/bin/env python3
"""VMM chunk-pool cost on one GPU: map / set-access / unmap latency per chunk,
bundles the size of a KV page, a qwen3.8 state slot and a K3 state, first
touch after mapping, and the jitter of a bandwidth loop while another thread
maps and unmaps.  usage: vmm_bench.py [gpu]"""
import ctypes as C
import statistics as st
import sys
import threading
import time

cu = C.CDLL("libcuda.so.1")


def chk(r, what):
    if r != 0:
        p = C.c_char_p()
        cu.cuGetErrorString(r, C.byref(p))
        raise RuntimeError(f"{what}: {r} {p.value}")


class Loc(C.Structure):
    _fields_ = [("type", C.c_int), ("id", C.c_int)]


class Flags(C.Structure):
    _fields_ = [("compressionType", C.c_ubyte), ("gpuDirectRDMACapable", C.c_ubyte), ("usage", C.c_ushort), ("reserved", C.c_ubyte * 4)]


class Prop(C.Structure):
    _fields_ = [("type", C.c_int), ("requestedHandleTypes", C.c_int), ("location", Loc), ("win32HandleMetaData", C.c_void_p), ("allocFlags", Flags)]


class Access(C.Structure):
    _fields_ = [("location", Loc), ("flags", C.c_int)]


cu.cuMemGetAllocationGranularity.argtypes = [C.POINTER(C.c_size_t), C.POINTER(Prop), C.c_int]
cu.cuMemAddressReserve.argtypes = [C.POINTER(C.c_ulonglong), C.c_size_t, C.c_size_t, C.c_ulonglong, C.c_ulonglong]
cu.cuMemCreate.argtypes = [C.POINTER(C.c_ulonglong), C.c_size_t, C.POINTER(Prop), C.c_ulonglong]
cu.cuMemMap.argtypes = [C.c_ulonglong, C.c_size_t, C.c_size_t, C.c_ulonglong, C.c_ulonglong]
cu.cuMemSetAccess.argtypes = [C.c_ulonglong, C.c_size_t, C.POINTER(Access), C.c_size_t]
cu.cuMemUnmap.argtypes = [C.c_ulonglong, C.c_size_t]
cu.cuMemRelease.argtypes = [C.c_ulonglong]
cu.cuMemsetD8Async.argtypes = [C.c_ulonglong, C.c_ubyte, C.c_size_t, C.c_void_p]
cu.cuMemcpyDtoDAsync_v2.argtypes = [C.c_ulonglong, C.c_ulonglong, C.c_size_t, C.c_void_p]
cu.cuMemAlloc_v2.argtypes = [C.POINTER(C.c_ulonglong), C.c_size_t]
cu.cuEventElapsedTime.argtypes = [C.POINTER(C.c_float), C.c_void_p, C.c_void_p]
cu.cuEventRecord.argtypes = [C.c_void_p, C.c_void_p]
cu.cuEventSynchronize.argtypes = [C.c_void_p]
cu.cuEventCreate.argtypes = [C.POINTER(C.c_void_p), C.c_uint]
cu.cuStreamCreate.argtypes = [C.POINTER(C.c_void_p), C.c_uint]
cu.cuStreamSynchronize.argtypes = [C.c_void_p]
cu.cuDevicePrimaryCtxRetain.argtypes = [C.POINTER(C.c_void_p), C.c_int]
cu.cuCtxSetCurrent.argtypes = [C.c_void_p]

MIB = 1 << 20
dev = int(sys.argv[1]) if len(sys.argv) > 1 else 0
chk(cu.cuInit(0), "cuInit")
ctx = C.c_void_p()
chk(cu.cuDevicePrimaryCtxRetain(C.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
chk(cu.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
prop = Prop()
prop.type = 1  # pinned
prop.requestedHandleTypes = 0
prop.location = Loc(1, dev)  # device
access = Access(Loc(1, dev), 3)  # read-write
now = time.perf_counter_ns

g = C.c_size_t()
chk(cu.cuMemGetAllocationGranularity(C.byref(g), C.byref(prop), 0), "granularity min")
gmin = g.value
chk(cu.cuMemGetAllocationGranularity(C.byref(g), C.byref(prop), 1), "granularity recommended")
print(f"granularity: min {gmin // MIB} MiB, recommended {g.value // MIB} MiB")


def event():
    e = C.c_void_p()
    chk(cu.cuEventCreate(C.byref(e), 0), "cuEventCreate")
    return e


def elapsed_ms(a, b):
    f = C.c_float()
    chk(cu.cuEventElapsedTime(C.byref(f), a, b), "cuEventElapsedTime")
    return f.value


stream = C.c_void_p()
chk(cu.cuStreamCreate(C.byref(stream), 1), "cuStreamCreate")


def create(size):
    h = C.c_ulonglong()
    chk(cu.cuMemCreate(C.byref(h), size, C.byref(prop), 0), "cuMemCreate")
    return h.value


def reserve(size, align):
    va = C.c_ulonglong()
    chk(cu.cuMemAddressReserve(C.byref(va), size, align, 0, 0), "cuMemAddressReserve")
    return va.value


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def fmt(xs):
    return f"p50 {q(xs, 0.5):8.1f}  p90 {q(xs, 0.9):8.1f}  max {max(xs):8.1f} us"


def bundle(chunk, n, reps=12, label=""):
    """Map n chunks contiguously (one slot / page), set access once, first touch, unmap."""
    t = now()
    handles = [create(chunk) for _ in range(n)]
    create_us = (now() - t) / 1e3 / n
    va = reserve(chunk * n, chunk)
    m, s, u1, ua, tot, first, second = [], [], [], [], [], [], []
    for r in range(reps):
        t0 = now()
        for i, h in enumerate(handles):
            t = now()
            chk(cu.cuMemMap(va + i * chunk, chunk, 0, h, 0), "cuMemMap")
            m.append((now() - t) / 1e3)
        t = now()
        chk(cu.cuMemSetAccess(va, chunk * n, C.byref(access), 1), "cuMemSetAccess")
        s.append((now() - t) / 1e3)
        tot.append((now() - t0) / 1e3)
        a, b = event(), event()
        chk(cu.cuEventRecord(a, stream), "rec")
        chk(cu.cuMemsetD8Async(va, 0, chunk * n, stream), "memset")
        chk(cu.cuEventRecord(b, stream), "rec")
        chk(cu.cuEventSynchronize(b), "sync")
        first.append(elapsed_ms(a, b) * 1e3)
        chk(cu.cuEventRecord(a, stream), "rec")
        chk(cu.cuMemsetD8Async(va, 1, chunk * n, stream), "memset")
        chk(cu.cuEventRecord(b, stream), "rec")
        chk(cu.cuEventSynchronize(b), "sync")
        second.append(elapsed_ms(a, b) * 1e3)
        if r % 2 == 0:
            for i in range(n):
                t = now()
                chk(cu.cuMemUnmap(va + i * chunk, chunk), "cuMemUnmap")
                u1.append((now() - t) / 1e3)
        else:
            t = now()
            chk(cu.cuMemUnmap(va, chunk * n), "cuMemUnmap")
            ua.append((now() - t) / 1e3)
    t = now()
    for h in handles:
        chk(cu.cuMemRelease(h), "cuMemRelease")
    release_us = (now() - t) / 1e3 / n
    print(f"\n{label}: {n} x {chunk // MIB} MiB = {chunk * n / MIB:.0f} MiB   (cuMemCreate {create_us:.1f} us/chunk, cuMemRelease {release_us:.1f} us/chunk)")
    print(f"  cuMemMap per chunk      {fmt(m)}")
    print(f"  cuMemSetAccess (bundle) {fmt(s)}")
    print(f"  bundle map+access total {fmt(tot)}")
    print(f"  cuMemUnmap per chunk    {fmt(u1)}")
    print(f"  cuMemUnmap whole bundle {fmt(ua)}")
    print(f"  memset first touch      {fmt(first)}   second {fmt(second)}")


for chunk, n, label in [
    (2 * MIB, 25, "KV page, 800 tok qwen3.8"),
    (2 * MIB, 74, "state slot qwen3.8"),
    (2 * MIB, 289, "state K3"),
    (32 * MIB, 5, "state slot qwen3.8 in 32 MiB chunks"),
    (32 * MIB, 19, "state K3 in 32 MiB chunks"),
]:
    bundle(chunk, n, label=label)


# Jitter of a bandwidth loop while another thread maps / unmaps slots.
copy_bytes = 1 << 30
src, dst = C.c_ulonglong(), C.c_ulonglong()
chk(cu.cuMemAlloc_v2(C.byref(src), copy_bytes), "cuMemAlloc")
chk(cu.cuMemAlloc_v2(C.byref(dst), copy_bytes), "cuMemAlloc")


def bw_loop(iters):
    evs = [(event(), event()) for _ in range(iters)]
    for a, b in evs:
        chk(cu.cuEventRecord(a, stream), "rec")
        chk(cu.cuMemcpyDtoDAsync_v2(dst.value, src.value, copy_bytes, stream), "memcpy")
        chk(cu.cuEventRecord(b, stream), "rec")
    chk(cu.cuStreamSynchronize(stream), "sync")
    return [elapsed_ms(a, b) * 1e3 for a, b in evs]


def mapper(chunk, n, stop, count):
    chk(cu.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
    handles = [create(chunk) for _ in range(n)]
    va = reserve(chunk * n, chunk)
    while not stop.is_set():
        for i, h in enumerate(handles):
            chk(cu.cuMemMap(va + i * chunk, chunk, 0, h, 0), "cuMemMap")
        chk(cu.cuMemSetAccess(va, chunk * n, C.byref(access), 1), "cuMemSetAccess")
        chk(cu.cuMemUnmap(va, chunk * n), "cuMemUnmap")
        count[0] += 1
    for h in handles:
        chk(cu.cuMemRelease(h), "cuMemRelease")


bw_loop(50)
base = bw_loop(400)
print(f"\nbandwidth loop alone ({copy_bytes // MIB} MiB DtoD x 400): {fmt(base)}  ({2 * copy_bytes / q(base, 0.5) / 1e6:.2f} TB/s at p50)")
for chunk, n, label in [(2 * MIB, 74, "qwen3.8 slots"), (2 * MIB, 289, "K3 states")]:
    stop, count = threading.Event(), [0]
    th = threading.Thread(target=mapper, args=(chunk, n, stop, count))
    th.start()
    time.sleep(0.05)
    t0 = now()
    under = bw_loop(400)
    dt = (now() - t0) / 1e9
    stop.set()
    th.join()
    print(f"bandwidth loop while mapping {label} ({count[0] / dt:.0f} map+unmap cycles/s): {fmt(under)}")
