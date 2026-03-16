#!/usr/bin/env python3
"""
店舗回転率シミュレーター
MaxValu 店舗レイアウトに基づく 100 人の客の行動シミュレーション。

技術スタック:
  - Streamlit : UI / アニメーション (st.empty)
  - Pymunk    : 物理演算（エージェント衝突・壁衝突）
  - Pillow    : フレームレンダリング
"""

import math
import os
import random
import time

import numpy as np
import pandas as pd
import pymunk
import streamlit as st
from PIL import Image, ImageDraw

# ──────────────────────────────────────────────────────────────────────────────
# Page config  (must be the very first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="店舗回転率シミュレーター",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Simulation constants
# ──────────────────────────────────────────────────────────────────────────────
IMG_W = 800          # floor-plan width  (px)
IMG_H = 600          # floor-plan height (px)
NUM_AGENTS = 100
NUM_REG = 5          # number of cash registers
AGENT_R        = 7     # agent circle radius (px)
BASE_SPEED     = 90    # px / simulation-second
TARGET_FPS     = 20    # target rendering frame rate (actual rate may vary with load)
ENTRY_INTERVAL = 0.4   # seconds between each agent's entry (sim-time)
OFFSCREEN_POS  = (-300, -300)   # position used to park off-screen agents
FLOOR_PLAN_FILE = "floor_plan.jpg"

# ──────────────────────────────────────────────────────────────────────────────
# Agent states
# ──────────────────────────────────────────────────────────────────────────────
S_WAITING   = 0   # has not entered yet
S_ENTERING  = 1   # walking toward sale area
S_AT_SALE   = 2   # dwelling at sale area  (5 s)
S_TO_SHELF1 = 3   # walking to first random shelf
S_AT_SHELF1 = 4   # dwelling at shelf 1   (3 s)
S_TO_SHELF2 = 5   # walking to second random shelf
S_AT_SHELF2 = 6   # dwelling at shelf 2   (3 s)
S_TO_REG    = 7   # walking to assigned register
S_IN_QUEUE  = 8   # queued, waiting turn
S_CHECKOUT  = 9   # being served          (3–8 s)
S_EXITING   = 10  # walking to exit
S_DONE      = 11  # left the store

# Colours used when drawing each state (R, G, B)
STATE_COLORS = {
    S_WAITING:   (190, 190, 190),
    S_ENTERING:  (60,  200, 60),
    S_AT_SALE:   (255, 140, 0),
    S_TO_SHELF1: (90,  130, 220),
    S_AT_SHELF1: (50,  70,  220),
    S_TO_SHELF2: (150, 90,  210),
    S_AT_SHELF2: (110, 50,  200),
    S_TO_REG:    (215, 55,  55),
    S_IN_QUEUE:  (195, 75,  75),
    S_CHECKOUT:  (255, 30,  30),
    S_EXITING:   (140, 215, 140),
    S_DONE:      (90,  90,  90),
}

# ──────────────────────────────────────────────────────────────────────────────
# Layout helpers
# ──────────────────────────────────────────────────────────────────────────────
def _shelf_rects(img_w: int = IMG_W, img_h: int = IMG_H):
    """Return shelf bounding boxes as (x1, y1, x2, y2) tuples."""
    shelf_h = 16
    aisle_h = 28
    ml = 45                    # left margin (px)
    mr = img_w - 45            # right margin end (px)
    cx = img_w // 2
    cg = 70                    # centre-aisle gap width (px)
    rects = []
    for ry in [65, 175, 280, 385]:
        # Left bank – two parallel shelves
        rects += [
            (ml, ry,                       cx - cg // 2, ry + shelf_h),
            (ml, ry + shelf_h + aisle_h,   cx - cg // 2, ry + shelf_h * 2 + aisle_h),
        ]
        # Right bank
        rects += [
            (cx + cg // 2, ry,                       mr, ry + shelf_h),
            (cx + cg // 2, ry + shelf_h + aisle_h,   mr, ry + shelf_h * 2 + aisle_h),
        ]
    return rects


def compute_locations(sale_xf: float = 0.50, sale_yf: float = 0.30):
    """
    Return the key simulation waypoints as pixel coordinates.

    Returns
    -------
    entrance  : (x, y)  – bottom-right gap in the outer wall
    exit_pt   : (x, y)  – bottom-left gap
    sale_area : (x, y)  – special-offer display (adjustable)
    registers : list of (x, y) for each of NUM_REG cash registers
    shelves   : list of (x, y) browse waypoints inside the aisles
    """
    entrance  = (int(IMG_W * 0.80), int(IMG_H * 0.92))
    exit_pt   = (int(IMG_W * 0.20), int(IMG_H * 0.92))
    sale_area = (int(IMG_W * sale_xf), int(IMG_H * sale_yf))

    # 5 registers spread across the bottom-centre section
    reg_y  = int(IMG_H * 0.80)
    reg_xs = [int(IMG_W * (0.28 + i * 0.085)) for i in range(NUM_REG)]
    registers = [(x, reg_y) for x in reg_xs]

    # Browse spots located in the walkable aisles between shelf pairs
    shelves = []
    for col_frac in [0.10, 0.16, 0.23, 0.30, 0.36,
                     0.64, 0.70, 0.77, 0.84, 0.90]:
        for row_frac in [0.18, 0.35, 0.52]:
            shelves.append((int(IMG_W * col_frac), int(IMG_H * row_frac)))

    return entrance, exit_pt, sale_area, registers, shelves


# ──────────────────────────────────────────────────────────────────────────────
# Floor-plan image generation
# ──────────────────────────────────────────────────────────────────────────────
def generate_floor_plan(sale_xf: float = 0.50, sale_yf: float = 0.30) -> Image.Image:
    """
    Create a MaxValu-style floor plan and save it as floor_plan.jpg.

    Black pixels  → physical walls recognised by the physics engine.
    White/light   → walkable floor.
    """
    img  = Image.new("RGB", (IMG_W, IMG_H), (245, 242, 235))
    draw = ImageDraw.Draw(img)

    # Checkerboard floor tiles
    tile = 40
    for ty in range(0, IMG_H, tile):
        for tx in range(0, IMG_W, tile):
            shade = (222, 218, 210) if (tx // tile + ty // tile) % 2 == 0 \
                    else (230, 226, 218)
            draw.rectangle([tx, ty, tx + tile - 1, ty + tile - 1], fill=shade)

    wall_c  = (18, 18, 18)    # near-black outer walls
    shelf_c = (35, 35, 35)    # dark shelves
    wt      = 10               # wall thickness

    # ── Outer walls (with gaps for entrance and exit) ─────────────────────
    entrance_x = int(IMG_W * 0.80)
    exit_x     = int(IMG_W * 0.20)
    gap        = 26           # gap width for doors

    draw.rectangle([0, 0, IMG_W, wt], fill=wall_c)                            # top
    draw.rectangle([0, 0, wt, IMG_H], fill=wall_c)                            # left
    draw.rectangle([IMG_W - wt, 0, IMG_W, IMG_H], fill=wall_c)               # right
    # Bottom wall – three segments with two door gaps
    draw.rectangle([0,               IMG_H - wt, exit_x - gap,     IMG_H], fill=wall_c)
    draw.rectangle([exit_x + gap,    IMG_H - wt, entrance_x - gap, IMG_H], fill=wall_c)
    draw.rectangle([entrance_x + gap, IMG_H - wt, IMG_W,           IMG_H], fill=wall_c)

    # ── Shelving units (black = wall) ─────────────────────────────────────
    for x1, y1, x2, y2 in _shelf_rects(IMG_W, IMG_H):
        draw.rectangle([x1, y1, x2, y2], fill=shelf_c)

    # ── Cash-register counters ─────────────────────────────────────────────
    _, _, _, registers, _ = compute_locations(sale_xf, sale_yf)
    for rx, ry in registers:
        draw.rectangle(
            [rx - 18, ry - 10, rx + 18, ry + 10],
            fill=(55, 55, 160), outline=(180, 180, 255), width=2,
        )

    # ── Special-offer area marker ──────────────────────────────────────────
    sx, sy = int(IMG_W * sale_xf), int(IMG_H * sale_yf)
    draw.ellipse([sx - 34, sy - 34, sx + 34, sy + 34],
                 fill=(255, 215, 0), outline=(210, 95, 0), width=3)

    # ── Entrance / Exit door markers ───────────────────────────────────────
    draw.rectangle([entrance_x - 24, IMG_H - wt - 4, entrance_x + 24, IMG_H],
                   fill=(70, 185, 70))
    draw.rectangle([exit_x - 24, IMG_H - wt - 4, exit_x + 24, IMG_H],
                   fill=(195, 70, 70))

    # ── Text labels (outlined for readability without font files) ──────────
    for txt, cx, cy, fc in [
        ("入口", entrance_x, IMG_H - wt - 18, (0, 130, 0)),
        ("出口", exit_x,     IMG_H - wt - 18, (170, 0,  0)),
        ("特売", sx,         sy - 42,          (150, 75, 0)),
        ("レジ", int(IMG_W * 0.50), int(IMG_H * 0.80) - 22, (200, 200, 255)),
    ]:
        for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((cx - 8 + ddx, cy + ddy), txt, fill=(255, 255, 255))
        draw.text((cx - 8, cy), txt, fill=fc)

    img.save(FLOOR_PLAN_FILE, "JPEG", quality=95)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# Pymunk physics space
# ──────────────────────────────────────────────────────────────────────────────
_WALL_FILTER  = pymunk.ShapeFilter(categories=0b01, mask=0b11)
_AGENT_FILTER = pymunk.ShapeFilter(categories=0b10, mask=0b11)


def _add_segment(space: pymunk.Space, p1, p2, thickness: int = 1):
    seg = pymunk.Segment(space.static_body, p1, p2, thickness)
    seg.elasticity = 0.0
    seg.friction   = 0.9
    seg.filter     = _WALL_FILTER
    space.add(seg)


def _add_wall_box(space: pymunk.Space, x1, y1, x2, y2):
    _add_segment(space, (x1, y1), (x2, y1))
    _add_segment(space, (x2, y1), (x2, y2))
    _add_segment(space, (x2, y2), (x1, y2))
    _add_segment(space, (x1, y2), (x1, y1))


def build_space() -> pymunk.Space:
    """
    Build a Pymunk Space whose static walls are derived from the black-pixel
    regions of the generated floor-plan image (outer boundary + shelf outlines).
    """
    space         = pymunk.Space()
    space.gravity = (0, 0)
    space.damping = 0.75

    wt         = 10
    entrance_x = int(IMG_W * 0.80)
    exit_x     = int(IMG_W * 0.20)
    gap        = 26

    # Outer perimeter
    _add_segment(space, (0, 0),       (IMG_W, 0))        # top
    _add_segment(space, (0, 0),       (0, IMG_H))        # left
    _add_segment(space, (IMG_W, 0),   (IMG_W, IMG_H))    # right
    _add_segment(space, (0,           IMG_H), (exit_x - gap,      IMG_H))
    _add_segment(space, (exit_x + gap, IMG_H), (entrance_x - gap, IMG_H))
    _add_segment(space, (entrance_x + gap, IMG_H), (IMG_W,        IMG_H))

    # Shelf walls (boxes matching the black shelf pixels in the floor plan)
    for x1, y1, x2, y2 in _shelf_rects(IMG_W, IMG_H):
        _add_wall_box(space, x1, y1, x2, y2)

    return space


# ──────────────────────────────────────────────────────────────────────────────
# Agent class
# ──────────────────────────────────────────────────────────────────────────────
class Agent:
    """One customer with individual speed, state machine, and a Pymunk body."""

    def __init__(self, agent_id: int, space: pymunk.Space, entry_delay: float):
        self.id          = agent_id
        self.state       = S_WAITING
        self.entry_delay = entry_delay          # sim-seconds before this agent enters
        self.dwell_time  = 0.0                  # remaining dwell at current spot
        self.speed       = random.uniform(0.8, 1.2) * BASE_SPEED
        self.shelf1      = None                 # (x, y) of first browse spot
        self.shelf2      = None                 # (x, y) of second browse spot
        self.register_id = None
        self.queue_pos   = None
        self._target     = None                 # current movement waypoint

        # Pymunk dynamic body (circle)
        mass   = 1.0
        moment = pymunk.moment_for_circle(mass, 0, AGENT_R)
        self.body  = pymunk.Body(mass, moment)
        self.body.position = OFFSCREEN_POS      # parked off-screen
        self.shape = pymunk.Circle(self.body, AGENT_R)
        self.shape.elasticity = 0.25
        self.shape.friction   = 0.5
        self.shape.filter     = _AGENT_FILTER
        space.add(self.body, self.shape)

    # ── position helpers ─────────────────────────────────────────────────────
    @property
    def pos(self):
        return self.body.position

    def set_target(self, tgt):
        self._target = tgt

    def reached(self, threshold: int = 15) -> bool:
        if self._target is None:
            return False
        dx = self._target[0] - self.pos.x
        dy = self._target[1] - self.pos.y
        return dx * dx + dy * dy < threshold * threshold

    # ── steering ─────────────────────────────────────────────────────────────
    def steer(self):
        """
        Apply velocity toward the current waypoint.
        Agents that are dwelling or done have their velocity zeroed.
        Pymunk then resolves any overlap (collision) each step.
        """
        if self.state in (S_WAITING, S_AT_SALE, S_AT_SHELF1, S_AT_SHELF2,
                          S_CHECKOUT, S_DONE):
            self.body.velocity = (0, 0)
            return
        if self._target is None:
            self.body.velocity = (0, 0)
            return

        dx = self._target[0] - self.pos.x
        dy = self._target[1] - self.pos.y
        d  = math.hypot(dx, dy)
        if d < 1:
            self.body.velocity = (0, 0)
        else:
            scale = self.speed / d
            self.body.velocity = (dx * scale, dy * scale)


# ──────────────────────────────────────────────────────────────────────────────
# Simulation class
# ──────────────────────────────────────────────────────────────────────────────
class Simulation:
    """Holds the full simulation state: space, agents, queues, statistics."""

    def __init__(self, sale_xf: float, sale_yf: float):
        self.elapsed       = 0.0
        self.exit_count    = 0
        self.exit_history  = []          # [{"time": int, "exits": int}, ...]
        self._next_hist_t  = 10.0        # next recording time

        (self.entrance, self.exit_pt, self.sale_area,
         self.registers, self.shelves) = compute_locations(sale_xf, sale_yf)

        self.space           = build_space()
        self.register_queues = [[] for _ in range(NUM_REG)]  # list of agent IDs

        # Stagger entries: agent i enters after i * ENTRY_INTERVAL sim-seconds
        self.agents = [
            Agent(i, self.space, entry_delay=i * ENTRY_INTERVAL)
            for i in range(NUM_AGENTS)
        ]

    # ── internal helpers ─────────────────────────────────────────────────────
    def _least_busy_reg(self) -> int:
        return min(range(NUM_REG), key=lambda r: len(self.register_queues[r]))

    @staticmethod
    def _queue_target(registers, reg_id: int, pos_in_queue: int):
        rx, ry = registers[reg_id]
        return (rx, ry + 24 + pos_in_queue * 18)

    def _agent(self, aid: int):
        return self.agents[aid] if 0 <= aid < len(self.agents) else None

    # ── per-agent state-machine update ───────────────────────────────────────
    def _update_agent(self, ag: Agent, dt: float):
        s = ag.state

        if s == S_WAITING:
            ag.entry_delay -= dt
            if ag.entry_delay <= 0:
                ex, ey = self.entrance
                ag.body.position = (ex + random.randint(-14, 14),
                                    ey + random.randint(-8, 6))
                ag.body.velocity = (0, 0)
                ag.set_target(self.sale_area)
                ag.state = S_ENTERING

        elif s == S_ENTERING:
            if ag.reached():
                ag.state      = S_AT_SALE
                ag.dwell_time = 5.0

        elif s == S_AT_SALE:
            ag.dwell_time -= dt
            if ag.dwell_time <= 0:
                s1, s2   = random.sample(self.shelves, 2)
                ag.shelf1 = s1
                ag.shelf2 = s2
                ag.set_target(s1)
                ag.state = S_TO_SHELF1

        elif s == S_TO_SHELF1:
            if ag.reached():
                ag.state      = S_AT_SHELF1
                ag.dwell_time = 3.0

        elif s == S_AT_SHELF1:
            ag.dwell_time -= dt
            if ag.dwell_time <= 0:
                ag.set_target(ag.shelf2)
                ag.state = S_TO_SHELF2

        elif s == S_TO_SHELF2:
            if ag.reached():
                ag.state      = S_AT_SHELF2
                ag.dwell_time = 3.0

        elif s == S_AT_SHELF2:
            ag.dwell_time -= dt
            if ag.dwell_time <= 0:
                reg_id         = self._least_busy_reg()
                qi             = len(self.register_queues[reg_id])
                ag.register_id = reg_id
                ag.queue_pos   = qi
                self.register_queues[reg_id].append(ag.id)
                ag.set_target(self._queue_target(self.registers, reg_id, qi))
                ag.state = S_TO_REG

        elif s == S_TO_REG:
            if ag.reached(threshold=14):
                ag.state = S_IN_QUEUE

        elif s == S_IN_QUEUE:
            if ag.queue_pos == 0:
                ag.state      = S_CHECKOUT
                ag.dwell_time = random.uniform(3.0, 8.0)

        elif s == S_CHECKOUT:
            ag.dwell_time -= dt
            if ag.dwell_time <= 0:
                rid = ag.register_id
                if ag.id in self.register_queues[rid]:
                    self.register_queues[rid].remove(ag.id)
                ag.register_id = None
                ag.queue_pos   = None
                ag.set_target(self.exit_pt)
                ag.state = S_EXITING

        elif s == S_EXITING:
            if ag.reached(threshold=22):
                ag.body.velocity = (0, 0)
                ag.body.position = OFFSCREEN_POS
                ag.state = S_DONE
                self.exit_count += 1

    # ── main simulation step ──────────────────────────────────────────────────
    def step(self, dt: float):
        """Advance the simulation by dt simulation-seconds."""
        for ag in self.agents:
            self._update_agent(ag, dt)
            ag.steer()

        # Sync queue positions so agents know their place in line
        for reg_id in range(NUM_REG):
            for qi, aid in enumerate(self.register_queues[reg_id]):
                ag = self._agent(aid)
                if ag is not None:
                    ag.queue_pos = qi
                    if ag.state in (S_TO_REG, S_IN_QUEUE):
                        ag.set_target(
                            self._queue_target(self.registers, reg_id, qi)
                        )

        self.space.step(dt)
        self.elapsed += dt

        # Record exit count every 10 sim-seconds for the chart
        if self.elapsed >= self._next_hist_t:
            self.exit_history.append(
                {"time": int(self.elapsed), "exits": self.exit_count}
            )
            self._next_hist_t += 10.0

    # ── statistics ───────────────────────────────────────────────────────────
    def in_store(self) -> int:
        """Number of customers currently inside the store."""
        return sum(
            1 for ag in self.agents
            if ag.state not in (S_WAITING, S_DONE)
        )

    def all_done(self) -> bool:
        return self.exit_count >= NUM_AGENTS


# ──────────────────────────────────────────────────────────────────────────────
# Frame renderer
# ──────────────────────────────────────────────────────────────────────────────
def render_frame(floor_img: Image.Image, sim: Simulation) -> Image.Image:
    """Draw the current simulation state onto a copy of the floor-plan image."""
    frame = floor_img.copy()
    draw  = ImageDraw.Draw(frame)

    # Queue indicator lines behind each register
    for reg_id, (rx, ry) in enumerate(sim.registers):
        n = len(sim.register_queues[reg_id])
        if n > 0:
            y_tail = Simulation._queue_target(sim.registers, reg_id, n - 1)[1]
            draw.line([(rx, ry + 12), (rx, y_tail + 10)],
                      fill=(255, 195, 80), width=3)

    # Agent circles
    for ag in sim.agents:
        if ag.state in (S_WAITING, S_DONE):
            continue
        px, py = int(ag.pos.x), int(ag.pos.y)
        color  = STATE_COLORS[ag.state]
        r      = AGENT_R
        draw.ellipse([px - r, py - r, px + r, py + r],
                     fill=color, outline=(15, 15, 15), width=1)

    # Queue-depth counters above each register
    for reg_id, (rx, ry) in enumerate(sim.registers):
        lbl = str(len(sim.register_queues[reg_id]))
        for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((rx - 4 + ddx, ry - 24 + ddy), lbl, fill=(0, 0, 0))
        draw.text((rx - 4, ry - 24), lbl, fill=(255, 240, 80))

    return frame


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────
st.title("🛒 店舗回転率シミュレーター")
st.caption(
    "MaxValu 店舗レイアウトに基づいた 100 人の客の行動シミュレーション。"
    "物理演算: Pymunk ／ UI: Streamlit"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 設定")

    sim_speed = st.slider("シミュレーション速度", 1, 5, 1,
                          help="1x = リアルタイム / 5x = 5倍速")

    st.markdown("---")
    st.subheader("特売エリア座標")
    sale_xf = st.slider("X 位置（左 0.0 ← → 右 1.0）", 0.10, 0.90, 0.50, 0.05)
    sale_yf = st.slider("Y 位置（上 0.0 ↓ → 下 0.70）", 0.10, 0.70, 0.30, 0.05)

    st.markdown("---")
    st.markdown("**エージェント凡例**")
    legend = [
        ("🟢", "入店中"),
        ("🟠", "特売エリア滞在"),
        ("🔵", "棚閲覧中"),
        ("🔴", "レジ列 / 会計"),
        ("🟩", "退店中"),
    ]
    for icon, label in legend:
        st.markdown(f"{icon} {label}")

    st.markdown("---")
    start_btn = st.button("▶ シミュレーション開始", use_container_width=True)
    reset_btn = st.button("🔄 リセット",           use_container_width=True)

# ── Top metric placeholders ───────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
ph_time   = col1.empty()
ph_inside = col2.empty()
ph_exits  = col3.empty()

# ── Simulation canvas ─────────────────────────────────────────────────────────
canvas = st.empty()

# ── Bottom chart placeholder ──────────────────────────────────────────────────
st.subheader("📈 時間ごとの累計退店数")
chart_ph = st.empty()

# ── Session-state initialisation ─────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False
if "sim" not in st.session_state:
    st.session_state.sim = None

if reset_btn:
    st.session_state.running = False
    st.session_state.sim     = None
    # Remove cached floor plan so it is regenerated with current sale coords
    if os.path.exists(FLOOR_PLAN_FILE):
        os.remove(FLOOR_PLAN_FILE)

if start_btn:
    st.session_state.running = True
    st.session_state.sim     = None          # force re-initialisation


# ──────────────────────────────────────────────────────────────────────────────
# Helper: load or (re-)generate the floor-plan image
# ──────────────────────────────────────────────────────────────────────────────
def _load_floor_plan(sale_xf: float, sale_yf: float) -> Image.Image:
    if not os.path.exists(FLOOR_PLAN_FILE):
        return generate_floor_plan(sale_xf, sale_yf)
    return Image.open(FLOOR_PLAN_FILE)


# ──────────────────────────────────────────────────────────────────────────────
# Idle display (before simulation starts)
# ──────────────────────────────────────────────────────────────────────────────
if not st.session_state.running:
    floor_img = _load_floor_plan(sale_xf, sale_yf)
    canvas.image(floor_img, use_container_width=True)
    ph_time.metric("⏱ 経過時間", "00:00")
    ph_inside.metric("🛒 店内人数", 0)
    ph_exits.metric("✅ 累計退店", 0)
    st.info("👈 サイドバーの **▶ シミュレーション開始** を押してください。")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Simulation initialisation (first run after Start button)
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.sim is None:
    # Regenerate the floor plan with the current sale-area position
    if os.path.exists(FLOOR_PLAN_FILE):
        os.remove(FLOOR_PLAN_FILE)
    floor_img = generate_floor_plan(sale_xf, sale_yf)
    st.session_state.floor_img = floor_img
    st.session_state.sim       = Simulation(sale_xf, sale_yf)

sim:       Simulation  = st.session_state.sim
floor_img: Image.Image = st.session_state.floor_img

# ──────────────────────────────────────────────────────────────────────────────
# Animation loop
# ──────────────────────────────────────────────────────────────────────────────
dt_real = 1.0 / TARGET_FPS  # wall-clock seconds per frame

while st.session_state.get("running", False):
    t0 = time.time()

    # How many physics sub-steps to run at current speed multiplier
    # We advance the simulation by (dt_real * sim_speed) sim-seconds,
    # split into sim_speed sub-steps each of dt_real sim-seconds to keep
    # physics stable.
    for _ in range(sim_speed):
        sim.step(dt_real)

    # ── Render frame ────────────────────────────────────────────────────────
    frame = render_frame(floor_img, sim)
    canvas.image(frame, use_container_width=True)

    # ── Update top metrics ───────────────────────────────────────────────────
    mins  = int(sim.elapsed) // 60
    secs  = int(sim.elapsed) % 60
    ph_time.metric("⏱ 経過時間",    f"{mins:02d}:{secs:02d}")
    ph_inside.metric("🛒 店内人数",  sim.in_store())
    ph_exits.metric("✅ 累計退店",   sim.exit_count)

    # ── Update exit chart ────────────────────────────────────────────────────
    if sim.exit_history:
        df = pd.DataFrame(sim.exit_history).set_index("time")
        chart_ph.line_chart(df["exits"], height=200)

    # ── Termination check ────────────────────────────────────────────────────
    if sim.all_done():
        mins = int(sim.elapsed) // 60
        secs = int(sim.elapsed) % 60
        canvas.success(
            f"✅ シミュレーション完了！  "
            f"全 {NUM_AGENTS} 人が退店しました。  "
            f"（シミュレーション時間 {mins:02d}:{secs:02d}）"
        )
        break

    # ── Frame-rate throttle ──────────────────────────────────────────────────
    elapsed_wall = time.time() - t0
    sleep_s      = max(0.0, dt_real - elapsed_wall)
    time.sleep(sleep_s)
