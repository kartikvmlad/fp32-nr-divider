# SPDX-License-Identifier: Apache-2.0
import os
import sys
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from fp32_div_model import fp32_div  # noqa: E402

WR, START = 0x04, 0x08
TIMEOUT = 300          # clocks; a divide takes 122


def pins(addr):
    """uio_in value for a 3-bit register address: {uio[4], uio[1:0]}."""
    return ((addr & 4) << 2) | (addr & 3)


async def reset(dut):
    clock = Clock(dut.clk, 20, unit="ns")       # 50 MHz
    cocotb.start_soon(clock.start())
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


async def write_byte(dut, addr, data):
    dut.ui_in.value = data
    dut.uio_in.value = WR | pins(addr)
    await ClockCycles(dut.clk, 3)
    dut.uio_in.value = pins(addr)
    await ClockCycles(dut.clk, 1)


async def read_byte(dut, addr):
    dut.uio_in.value = pins(addr)
    await ClockCycles(dut.clk, 1)
    return int(dut.uo_out.value)


async def write_operands(dut, a, b):
    for i in range(4):
        await write_byte(dut, i, (a >> (8 * i)) & 0xFF)
    for i in range(4):
        await write_byte(dut, 4 + i, (b >> (8 * i)) & 0xFF)


async def pulse_start(dut):
    dut.uio_in.value = START
    await ClockCycles(dut.clk, 3)
    dut.uio_in.value = 0


async def wait_done(dut, what=""):
    for _ in range(TIMEOUT):
        await ClockCycles(dut.clk, 1)
        if int(dut.uio_out.value) & 0x80:
            return
    raise AssertionError(f"timeout {what}")


async def read_result(dut):
    q = 0
    for i in range(4):
        q |= (await read_byte(dut, i)) << (8 * i)
    flags = await read_byte(dut, 4)
    return q, flags


async def divide(dut, a, b):
    await write_operands(dut, a, b)
    await pulse_start(dut)
    await wait_done(dut, f"a={a:08X} b={b:08X}")
    return await read_result(dut)


async def check(dut, a, b):
    q, flags = await divide(dut, a, b)
    er, ef = fp32_div(a, b)
    er, ef = int(er), int(ef)
    assert (q, flags) == (er, ef), \
        f"{a:08X}/{b:08X}: got {q:08X} flags {flags:08b}, expected {er:08X} flags {ef:05b}"


CORNERS = [0x00000000, 0x80000000, 0x00000001, 0x007FFFFF, 0x00800000, 0x3F800000, 0xBF800000,
           0x40000000, 0x3EAAAAAB, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000,
           0x7FA00000, 0x00400000, 0x10000000, 0x64000000, 0x3F7FFFFF, 0x3F800001]


@cocotb.test()
async def test_basic(dut):
    await reset(dut)
    dut._log.info("1.0 / 2.0 = 0.5")
    q, flags = await divide(dut, 0x3F800000, 0x40000000)
    assert q == 0x3F000000 and flags == 0
    dut._log.info("1.0 / 3.0 = 0x3EAAAAAB, inexact")
    q, flags = await divide(dut, 0x3F800000, 0x40400000)
    assert q == 0x3EAAAAAB and flags == 0b00001


@cocotb.test()
async def test_corners(dut):
    await reset(dut)
    for a in CORNERS:
        for b in CORNERS:
            await check(dut, a, b)
    dut._log.info(f"{len(CORNERS) ** 2} corner-case pairs passed")


@cocotb.test()
async def test_random(dut):
    await reset(dut)
    rng = random.Random(2026)
    n = int(os.environ.get("N_RANDOM", "1000"))
    for _ in range(n):
        await check(dut, rng.getrandbits(32), rng.getrandbits(32))
    dut._log.info(f"{n} random pairs passed")


@cocotb.test()
async def test_write_while_busy(dut):
    """Operands are frozen during a divide; a strobe held across it lands afterwards."""
    await reset(dut)
    a, b, a2 = 0x45671234, 0x3A31BEEF, 0x12
    await write_operands(dut, a, b)
    dut.uio_in.value = START
    await ClockCycles(dut.clk, 3)
    # divide is running: present a new A low byte and hold the strobe until well after done
    dut.ui_in.value = a2
    dut.uio_in.value = WR | pins(0)
    await wait_done(dut)
    await ClockCycles(dut.clk, 3)
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, 1)
    q, flags = await read_result(dut)
    er, ef = fp32_div(a, b)
    assert (q, flags) == (int(er), int(ef)), "operands changed during the divide"
    # the held write must have landed once the divide finished
    await pulse_start(dut)
    await wait_done(dut)
    q, _ = await read_result(dut)
    er, _ = fp32_div((a & 0xFFFFFF00) | a2, b)
    assert q == int(er), "write held across a divide was lost"


@cocotb.test()
async def test_start_while_busy(dut):
    """A start pulse during a divide is ignored and does not disturb it."""
    await reset(dut)
    a, b = 0x5A5A1234, 0x2B17FACE
    await write_operands(dut, a, b)
    await pulse_start(dut)
    await ClockCycles(dut.clk, 20)
    await pulse_start(dut)                      # second start, mid-divide
    # the stray start clears the sticky done flag, so wait on busy (uio_out[6]) instead
    for _ in range(TIMEOUT):
        await ClockCycles(dut.clk, 1)
        if not int(dut.uio_out.value) & 0x40:
            break
    await ClockCycles(dut.clk, 2)
    q, flags = await read_result(dut)
    er, ef = fp32_div(a, b)
    assert (q, flags) == (int(er), int(ef))


@cocotb.test()
async def test_subnormal(dut):
    await reset(dut)
    rng = random.Random(7)
    for _ in range(300):
        sub = rng.randrange(1, 1 << 23) | (rng.randrange(2) << 31)
        other = rng.getrandbits(32)
        await check(dut, sub, other)
        await check(dut, other, sub)
    dut._log.info("600 subnormal pairs passed")
