import json
import matplotlib.pyplot as plt
import numpy as np

# Read the JSON file
with open('data.json', 'r') as f:
    data = json.load(f)

# Extract acceleration data
accel_data = data[0]['data']
x = np.array(accel_data['x'])
y = np.array(accel_data['y'])
z = np.array(accel_data['z'])

# Create time axis (sample index)
time = np.arange(len(x))

# Create figure with subplots
fig, axes = plt.subplots(4, 1, figsize=(12, 10))
fig.suptitle('Acceleration Data - "correct" Label', fontsize=16, fontweight='bold')

# Plot X acceleration
axes[0].plot(time, x, 'r-', linewidth=0.8)
axes[0].set_ylabel('X Acceleration', fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, len(time))

# Plot Y acceleration
axes[1].plot(time, y, 'g-', linewidth=0.8)
axes[1].set_ylabel('Y Acceleration', fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, len(time))

# Plot Z acceleration
axes[2].plot(time, z, 'b-', linewidth=0.8)
axes[2].set_ylabel('Z Acceleration', fontsize=10)
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(0, len(time))

# Plot all three together
axes[3].plot(time, x, 'r-', linewidth=0.8, label='X', alpha=0.7)
axes[3].plot(time, y, 'g-', linewidth=0.8, label='Y', alpha=0.7)
axes[3].plot(time, z, 'b-', linewidth=0.8, label='Z', alpha=0.7)
axes[3].set_ylabel('All Axes', fontsize=10)
axes[3].set_xlabel('Sample Index', fontsize=10)
axes[3].legend(loc='upper right')
axes[3].grid(True, alpha=0.3)
axes[3].set_xlim(0, len(time))

plt.tight_layout()
plt.show()

# Optional: Create a 3D trajectory plot
fig2 = plt.figure(figsize=(10, 8))
ax = fig2.add_subplot(111, projection='3d')

# Plot the 3D trajectory
ax.plot(x, y, z, linewidth=0.8)
ax.scatter(x[0], y[0], z[0], c='green', marker='o', s=100, label='Start')
ax.scatter(x[-1], y[-1], z[-1], c='red', marker='x', s=100, label='End')

ax.set_xlabel('X Acceleration')
ax.set_ylabel('Y Acceleration')
ax.set_zlabel('Z Acceleration')
ax.set_title('3D Acceleration Trajectory - "correct" Label', fontweight='bold')
ax.legend()

plt.tight_layout()
plt.show()

# Print statistics
print(f"Data Statistics:")
print(f"Number of samples: {len(x)}")
print(f"\nX-axis: min={x.min():.3f}, max={x.max():.3f}, mean={x.mean():.3f}, std={x.std():.3f}")
print(f"Y-axis: min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}, std={y.std():.3f}")
print(f"Z-axis: min={z.min():.3f}, max={z.max():.3f}, mean={z.mean():.3f}, std={z.std():.3f}")