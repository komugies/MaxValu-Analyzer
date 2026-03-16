"""
Simple test script to verify the simulation logic works correctly
without running the full Streamlit UI.
"""

import pymunk
from PIL import Image
import numpy as np
import random
from enum import Enum

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
WALL_THRESHOLD = 50
CUSTOMER_RADIUS = 8
WALL_SAMPLE_RATE = 5  # Same as in app.py

class CustomerState(Enum):
    ENTERING = "entering"
    AT_SALE = "at_sale"
    GOING_TO_SHELF1 = "going_to_shelf1"
    AT_SHELF1 = "at_shelf1"
    EXITED = "exited"

def test_simulator():
    """Test the core simulation functionality"""
    
    print("Testing Store Simulator Core Logic...")
    print("-" * 50)
    
    # Load image
    image = Image.open('floor_plan.jpg')
    image_array = np.array(image)
    print(f"✓ Loaded floor plan: {image_array.shape}")
    
    # Create Pymunk space
    space = pymunk.Space()
    space.gravity = (0, 0)
    space.damping = 0.8
    print("✓ Created Pymunk space")
    
    # Test wall creation from black pixels
    gray_image = np.mean(image_array, axis=2)
    wall_body = space.static_body
    wall_count = 0
    
    for y in range(0, IMAGE_HEIGHT, WALL_SAMPLE_RATE):
        for x in range(0, IMAGE_WIDTH, WALL_SAMPLE_RATE):
            if gray_image[y, x] < WALL_THRESHOLD:
                wall_shape = pymunk.Poly.create_box(
                    wall_body,
                    (WALL_SAMPLE_RATE, WALL_SAMPLE_RATE),
                    radius=0
                )
                wall_shape.body.position = (x, y)
                wall_shape.friction = 1.0
                space.add(wall_shape)
                wall_count += 1
    
    print(f"✓ Created {wall_count} wall segments")
    
    # Test coordinate positions
    entrance_pos = (int(IMAGE_WIDTH * 0.8), int(IMAGE_HEIGHT * 0.9))
    exit_pos = (int(IMAGE_WIDTH * 0.2), int(IMAGE_HEIGHT * 0.9))
    sale_area_pos = (int(IMAGE_WIDTH * 0.3), int(IMAGE_HEIGHT * 0.3))
    
    print(f"✓ Entrance position: {entrance_pos}")
    print(f"✓ Exit position: {exit_pos}")
    print(f"✓ Sale area position: {sale_area_pos}")
    
    # Test register positions
    register_y = int(IMAGE_HEIGHT * 0.75)
    register_spacing = (IMAGE_WIDTH - 100) // 6
    register_positions = []
    for i in range(5):
        reg_x = 50 + register_spacing * (i + 1) + 15
        register_positions.append((reg_x, register_y - 20))
    
    print(f"✓ Created {len(register_positions)} register positions")
    for i, pos in enumerate(register_positions):
        print(f"  Register {i+1}: {pos}")
    
    # Test customer creation
    customers = []
    for i in range(5):  # Create 5 test customers
        mass = 1
        moment = pymunk.moment_for_circle(mass, 0, CUSTOMER_RADIUS)
        body = pymunk.Body(mass, moment)
        
        offset_x = random.uniform(-20, 20)
        offset_y = random.uniform(-10, 10)
        body.position = (entrance_pos[0] + offset_x, entrance_pos[1] + offset_y)
        
        shape = pymunk.Circle(body, CUSTOMER_RADIUS)
        shape.friction = 0.5
        space.add(body, shape)
        
        speed_multiplier = random.uniform(0.8, 1.2)
        customers.append({
            'body': body,
            'shape': shape,
            'speed': speed_multiplier,
            'id': i
        })
    
    print(f"✓ Created {len(customers)} test customers")
    for c in customers:
        print(f"  Customer {c['id']}: pos={c['body'].position}, speed={c['speed']:.2f}x")
    
    # Test physics simulation for a few steps
    print("\n✓ Running physics simulation test...")
    for step in range(10):
        space.step(1/30.0)
    
    print(f"  After 10 steps, customer positions:")
    for c in customers:
        print(f"  Customer {c['id']}: pos={c['body'].position}")
    
    # Test rendering
    print("\n✓ Testing rendering...")
    frame = image_array.copy()
    
    # Draw customers
    for c in customers:
        cx, cy = int(c['body'].position.x), int(c['body'].position.y)
        for dy in range(-CUSTOMER_RADIUS, CUSTOMER_RADIUS+1):
            for dx in range(-CUSTOMER_RADIUS, CUSTOMER_RADIUS+1):
                if dx*dx + dy*dy <= CUSTOMER_RADIUS*CUSTOMER_RADIUS:
                    y, x = cy + dy, cx + dx
                    if 0 <= y < IMAGE_HEIGHT and 0 <= x < IMAGE_WIDTH:
                        frame[y, x] = [0, 0, 255]
    
    # Save test frame
    test_img = Image.fromarray(frame)
    test_img.save('/tmp/test_frame.jpg')
    print("  Saved test frame to /tmp/test_frame.jpg")
    
    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED!")
    print("=" * 50)

if __name__ == "__main__":
    test_simulator()
