import streamlit as st
import pymunk
from PIL import Image
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

# Page configuration
st.set_page_config(page_title="店舗回転率シミュレーター", layout="wide")

# Constants
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
WALL_THRESHOLD = 50  # Pixels darker than this are considered walls
CUSTOMER_RADIUS = 8
WALL_SAMPLE_RATE = 5  # Sample rate for wall detection
FPS = 30

class CustomerState(Enum):
    ENTERING = "entering"
    GOING_TO_SALE = "going_to_sale"
    AT_SALE = "at_sale"
    GOING_TO_SHELF1 = "going_to_shelf1"
    AT_SHELF1 = "at_shelf1"
    GOING_TO_SHELF2 = "going_to_shelf2"
    AT_SHELF2 = "at_shelf2"
    GOING_TO_REGISTER = "going_to_register"
    AT_REGISTER = "at_register"
    EXITING = "exiting"
    EXITED = "exited"

@dataclass
class Customer:
    body: pymunk.Body
    shape: pymunk.Circle
    state: CustomerState
    target: Optional[Tuple[float, float]]
    speed_multiplier: float
    state_timer: float
    shelf_targets: List[Tuple[float, float]]
    assigned_register: Optional[int]
    register_service_time: float  # Store service time when entering register
    id: int

class StoreSimulator:
    def __init__(self, image_path: str, sale_area_pos: Tuple[float, float]):
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.image_array = np.array(self.image)
        self.width = IMAGE_WIDTH
        self.height = IMAGE_HEIGHT
        
        # Setup Pymunk space
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)  # No gravity for top-down view
        self.space.damping = 0.8  # Add damping to smooth movement
        
        # Coordinates (relative to image size)
        self.entrance_pos = (int(self.width * 0.8), int(self.height * 0.9))
        self.exit_pos = (int(self.width * 0.2), int(self.height * 0.9))
        
        # Sale area (adjustable)
        self.sale_area_pos = (int(self.width * sale_area_pos[0]), int(self.height * sale_area_pos[1]))
        
        # Register positions (5 registers in a row at bottom center)
        register_y = int(self.height * 0.75)
        register_spacing = (self.width - 100) // 6
        self.register_positions = []
        for i in range(5):
            reg_x = 50 + register_spacing * (i + 1) + 15
            self.register_positions.append((reg_x, register_y - 20))
        
        # Random shelf positions for customers to visit
        self.shelf_positions = [
            (150, 200), (250, 200), (550, 200), (650, 200),
            (150, 300), (250, 300), (550, 300), (650, 300),
            (150, 400), (250, 400)
        ]
        
        # Setup walls from image
        self._setup_walls()
        
        # Customers
        self.customers: List[Customer] = []
        self.customer_id_counter = 0
        self.exited_count = 0
        self.register_queues = [0, 0, 0, 0, 0]  # Queue length for each register
        
        # Statistics
        self.start_time = time.time()
        self.exit_history = []  # (time, cumulative_exits)
        
    def _setup_walls(self):
        """Create static walls based on black pixels in the image"""
        # Convert image to grayscale for easier threshold detection
        gray_image = np.mean(self.image_array, axis=2)
        
        # Create walls for black pixels
        wall_body = self.space.static_body
        
        # Sample walls at intervals to avoid too many shapes
        for y in range(0, self.height, WALL_SAMPLE_RATE):
            for x in range(0, self.width, WALL_SAMPLE_RATE):
                if gray_image[y, x] < WALL_THRESHOLD:
                    # Create a small square wall segment
                    wall_shape = pymunk.Poly.create_box(
                        wall_body,
                        (WALL_SAMPLE_RATE, WALL_SAMPLE_RATE),
                        radius=0
                    )
                    wall_shape.body.position = (x, y)
                    wall_shape.friction = 1.0
                    wall_shape.collision_type = 1  # Wall collision type
                    self.space.add(wall_shape)
    
    def create_customer(self) -> Customer:
        """Create a new customer at the entrance"""
        mass = 1
        moment = pymunk.moment_for_circle(mass, 0, CUSTOMER_RADIUS)
        body = pymunk.Body(mass, moment)
        
        # Add some random offset to entrance position to avoid overlap
        offset_x = random.uniform(-20, 20)
        offset_y = random.uniform(-10, 10)
        body.position = (self.entrance_pos[0] + offset_x, self.entrance_pos[1] + offset_y)
        
        shape = pymunk.Circle(body, CUSTOMER_RADIUS)
        shape.friction = 0.5
        shape.collision_type = 2  # Customer collision type
        
        self.space.add(body, shape)
        
        # Random speed multiplier (0.8x to 1.2x)
        speed_multiplier = random.uniform(0.8, 1.2)
        
        # Random shelf targets (2 random shelves)
        shelf_targets = random.sample(self.shelf_positions, 2)
        
        customer = Customer(
            body=body,
            shape=shape,
            state=CustomerState.ENTERING,
            target=self.sale_area_pos,
            speed_multiplier=speed_multiplier,
            state_timer=0,
            shelf_targets=shelf_targets,
            assigned_register=None,
            register_service_time=0.0,
            id=self.customer_id_counter
        )
        
        self.customer_id_counter += 1
        self.customers.append(customer)
        
        return customer
    
    def get_least_busy_register(self) -> int:
        """Find the register with the shortest queue"""
        return self.register_queues.index(min(self.register_queues))
    
    def move_towards_target(self, customer: Customer, base_speed: float = 50.0):
        """Move customer towards their target position"""
        if customer.target is None:
            return
        
        pos = customer.body.position
        target = customer.target
        
        # Calculate direction
        dx = target[0] - pos.x
        dy = target[1] - pos.y
        distance = (dx**2 + dy**2) ** 0.5
        
        if distance < 5:  # Close enough to target
            return
        
        # Normalize and apply speed
        speed = base_speed * customer.speed_multiplier
        vx = (dx / distance) * speed
        vy = (dy / distance) * speed
        
        customer.body.velocity = (vx, vy)
    
    def update_customer_state(self, customer: Customer, dt: float):
        """Update customer behavior based on their current state"""
        customer.state_timer += dt
        pos = customer.body.position
        
        # Check if reached target
        if customer.target:
            dx = customer.target[0] - pos.x
            dy = customer.target[1] - pos.y
            distance = (dx**2 + dy**2) ** 0.5
            at_target = distance < 15
        else:
            at_target = False
        
        # State machine
        if customer.state == CustomerState.ENTERING:
            self.move_towards_target(customer)
            if at_target:
                customer.state = CustomerState.AT_SALE
                customer.state_timer = 0
                customer.body.velocity = (0, 0)
        
        elif customer.state == CustomerState.AT_SALE:
            customer.body.velocity = (0, 0)
            if customer.state_timer >= 5.0:  # 5 seconds at sale area
                customer.state = CustomerState.GOING_TO_SHELF1
                customer.target = customer.shelf_targets[0]
                customer.state_timer = 0
        
        elif customer.state == CustomerState.GOING_TO_SHELF1:
            self.move_towards_target(customer)
            if at_target:
                customer.state = CustomerState.AT_SHELF1
                customer.state_timer = 0
                customer.body.velocity = (0, 0)
        
        elif customer.state == CustomerState.AT_SHELF1:
            customer.body.velocity = (0, 0)
            if customer.state_timer >= 3.0:  # 3 seconds at shelf
                customer.state = CustomerState.GOING_TO_SHELF2
                customer.target = customer.shelf_targets[1]
                customer.state_timer = 0
        
        elif customer.state == CustomerState.GOING_TO_SHELF2:
            self.move_towards_target(customer)
            if at_target:
                customer.state = CustomerState.AT_SHELF2
                customer.state_timer = 0
                customer.body.velocity = (0, 0)
        
        elif customer.state == CustomerState.AT_SHELF2:
            customer.body.velocity = (0, 0)
            if customer.state_timer >= 3.0:  # 3 seconds at shelf
                customer.state = CustomerState.GOING_TO_REGISTER
                # Assign to least busy register
                customer.assigned_register = self.get_least_busy_register()
                self.register_queues[customer.assigned_register] += 1
                customer.target = self.register_positions[customer.assigned_register]
                customer.state_timer = 0
        
        elif customer.state == CustomerState.GOING_TO_REGISTER:
            self.move_towards_target(customer)
            if at_target:
                customer.state = CustomerState.AT_REGISTER
                customer.state_timer = 0
                # Store service time when entering register state
                customer.register_service_time = random.uniform(2.0, 5.0)
                customer.body.velocity = (0, 0)
        
        elif customer.state == CustomerState.AT_REGISTER:
            customer.body.velocity = (0, 0)
            # Use stored service time
            if customer.state_timer >= customer.register_service_time:
                customer.state = CustomerState.EXITING
                if customer.assigned_register is not None:
                    self.register_queues[customer.assigned_register] -= 1
                customer.target = self.exit_pos
                customer.state_timer = 0
        
        elif customer.state == CustomerState.EXITING:
            self.move_towards_target(customer)
            # Check if exited (reached exit or left bounds)
            if at_target or pos.y > self.height:
                customer.state = CustomerState.EXITED
                self.exited_count += 1
                elapsed = time.time() - self.start_time
                self.exit_history.append((elapsed, self.exited_count))
    
    def update(self, dt: float):
        """Update simulation"""
        # Update all customers
        for customer in self.customers:
            if customer.state != CustomerState.EXITED:
                self.update_customer_state(customer, dt)
        
        # Step physics simulation
        self.space.step(dt)
        
        # Remove exited customers
        self.customers = [c for c in self.customers if c.state != CustomerState.EXITED]
    
    def render(self) -> np.ndarray:
        """Render the current simulation state"""
        # Create a copy of the background image
        frame = self.image_array.copy()
        
        # Draw sale area (yellow circle)
        sale_x, sale_y = self.sale_area_pos
        for dy in range(-15, 16):
            for dx in range(-15, 16):
                if dx*dx + dy*dy <= 15*15:
                    y, x = int(sale_y + dy), int(sale_x + dx)
                    if 0 <= y < self.height and 0 <= x < self.width:
                        frame[y, x] = [255, 255, 0]  # Yellow
        
        # Draw registers (green circles)
        for reg_x, reg_y in self.register_positions:
            for dy in range(-8, 9):
                for dx in range(-8, 9):
                    if dx*dx + dy*dy <= 8*8:
                        y, x = int(reg_y + dy), int(reg_x + dx)
                        if 0 <= y < self.height and 0 <= x < self.width:
                            frame[y, x] = [0, 255, 0]  # Green
        
        # Draw customers (blue circles)
        for customer in self.customers:
            if customer.state != CustomerState.EXITED:
                cx, cy = int(customer.body.position.x), int(customer.body.position.y)
                for dy in range(-CUSTOMER_RADIUS, CUSTOMER_RADIUS+1):
                    for dx in range(-CUSTOMER_RADIUS, CUSTOMER_RADIUS+1):
                        if dx*dx + dy*dy <= CUSTOMER_RADIUS*CUSTOMER_RADIUS:
                            y, x = cy + dy, cx + dx
                            if 0 <= y < self.height and 0 <= x < self.width:
                                frame[y, x] = [0, 0, 255]  # Blue
        
        return frame
    
    def get_current_customer_count(self) -> int:
        """Get number of customers currently in store"""
        return len([c for c in self.customers if c.state != CustomerState.EXITED])

def main():
    st.title("🏪 店舗回転率シミュレーター")
    st.markdown("**Streamlit + Pymunk による100人の客の流れシミュレーション**")
    
    # Sidebar controls
    st.sidebar.header("⚙️ 設定")
    
    # Simulation speed
    sim_speed = st.sidebar.slider(
        "シミュレーション速度",
        min_value=1.0,
        max_value=5.0,
        value=1.0,
        step=0.5,
        format="%.1fx"
    )
    
    # Sale area position
    st.sidebar.subheader("特売エリア位置")
    sale_x = st.sidebar.slider("X座標", min_value=0.1, max_value=0.9, value=0.3, step=0.05)
    sale_y = st.sidebar.slider("Y座標", min_value=0.1, max_value=0.6, value=0.3, step=0.05)
    
    # Initialize session state
    if 'simulator' not in st.session_state:
        st.session_state.simulator = StoreSimulator('floor_plan.jpg', (sale_x, sale_y))
        st.session_state.last_update = time.time()
        st.session_state.customers_created = 0
        st.session_state.running = False
    
    # Update sale area if changed
    current_sale_pos = (int(IMAGE_WIDTH * sale_x), int(IMAGE_HEIGHT * sale_y))
    if st.session_state.simulator.sale_area_pos != current_sale_pos:
        st.session_state.simulator.sale_area_pos = current_sale_pos
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶️ 開始" if not st.session_state.running else "⏸️ 一時停止"):
            st.session_state.running = not st.session_state.running
    with col2:
        if st.button("🔄 リセット"):
            st.session_state.simulator = StoreSimulator('floor_plan.jpg', (sale_x, sale_y))
            st.session_state.last_update = time.time()
            st.session_state.customers_created = 0
            st.session_state.running = False
    
    # Statistics area
    stat_cols = st.columns(3)
    with stat_cols[0]:
        elapsed_time = time.time() - st.session_state.simulator.start_time
        st.metric("⏱️ 経過時間", f"{elapsed_time:.1f}秒")
    with stat_cols[1]:
        current_count = st.session_state.simulator.get_current_customer_count()
        st.metric("🚶 現在の店内人数", f"{current_count}人")
    with stat_cols[2]:
        st.metric("✅ 累計退店数", f"{st.session_state.simulator.exited_count}人")
    
    # Main simulation area
    placeholder = st.empty()
    
    # Graph area
    graph_placeholder = st.empty()
    
    # Main loop
    if st.session_state.running:
        current_time = time.time()
        dt = (current_time - st.session_state.last_update) * sim_speed
        dt = min(dt, 0.1)  # Cap dt to avoid instability
        
        # Create new customers (spread over time)
        if st.session_state.customers_created < 100:
            # Add customers gradually (one every 0.5 seconds)
            if elapsed_time > st.session_state.customers_created * 0.5:
                st.session_state.simulator.create_customer()
                st.session_state.customers_created += 1
        
        # Update simulation
        if dt > 0:
            st.session_state.simulator.update(dt / FPS * 30)  # Normalize to expected frame rate
            st.session_state.last_update = current_time
        
        # Render
        frame = st.session_state.simulator.render()
        placeholder.image(frame, channels="RGB", use_container_width=True)
        
        # Update graph
        if len(st.session_state.simulator.exit_history) > 0:
            times, exits = zip(*st.session_state.simulator.exit_history)
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(times, exits, 'b-', linewidth=2)
            ax.set_xlabel('経過時間 (秒)')
            ax.set_ylabel('累計退店数')
            ax.set_title('時間経過ごとの退店数')
            ax.grid(True, alpha=0.3)
            graph_placeholder.pyplot(fig)
            plt.close()
        
        # Rerun to continue animation
        time.sleep(0.03)  # ~30 FPS
        st.rerun()
    else:
        # Show current state when paused
        frame = st.session_state.simulator.render()
        placeholder.image(frame, channels="RGB", use_container_width=True)
        
        # Show graph even when paused
        if len(st.session_state.simulator.exit_history) > 0:
            times, exits = zip(*st.session_state.simulator.exit_history)
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(times, exits, 'b-', linewidth=2)
            ax.set_xlabel('経過時間 (秒)')
            ax.set_ylabel('累計退店数')
            ax.set_title('時間経過ごとの退店数')
            ax.grid(True, alpha=0.3)
            graph_placeholder.pyplot(fig)
            plt.close()

if __name__ == "__main__":
    main()
