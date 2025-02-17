import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class GUITracker:

    def __init__(self, minX = 0, maxX = 1.2, minY = 0, maxY = 0.5, minZ = 0, maxZ = 1):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Initial BLE position
        self.x = 0
        self.y = 0
        self.z = 0

        self.ax.set_title("Real-time BLE Position Tracking")

        # Set labels
        self.ax.set_xlabel("X Axis")
        self.ax.set_ylabel("Z Axis") # y-axis <-> z-axis
        self.ax.set_zlabel("Y Axis") # y-axis <-> z-axis
        
        # Set range
        self.ax.set_xlim([minX, maxX])
        self.ax.set_ylim([minZ, maxZ])  # y-axis <-> z-axis
        self.ax.set_zlim([minY, maxY])  # y-axis <-> z-axis

        # Scatter plot
        self.scatter = self.ax.scatter([], [], [], c='r', marker='o', s=100)

        # Animation
        self.ani = FuncAnimation(self.fig, self._update_plot, interval=1000, blit=False)

    def update(self, x: float, y: float, z: float):
        """ Update BLE position """
        self.x = x
        self.y = y
        self.z = z

    def _update_plot(self, frame):
        """ Update plot data internally """
        self.scatter._offsets3d = ([self.x], [self.y], [self.z])
        return self.scatter

    def show(self):
        plt.show()

    def close(self):
        plt.close()