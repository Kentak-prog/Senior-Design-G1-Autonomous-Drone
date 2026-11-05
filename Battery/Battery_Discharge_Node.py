#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from px4_msgs.msg import ActuatorOutputs
import csv 

class DroneTheveninBatteryNode(Node):
    def __init__(self):
        super().__init__('drone_thevenin_battery_node')
        
        # 1. HARDWARE CONFIGURATION (Starling 2 Max)
        self.S_COUNT = 4              # 4S Configuration (Two 2S packs in series)
        self.CAPACITY_AH = 3.0        # Sony VTC6 capacity (3000 mAh)
        self.total_capacity_as = self.CAPACITY_AH * 3600.0 # Capacity in Amp-seconds
        self.remaining_capacity_as = self.total_capacity_as
        
        # Fixed Source & Load Resistances (Calculated outside the code)
        self.R_TH = 0.070             # 70 mOhms internal source resistance
        self.R_BOARD = 14.6           # VOXL 2 Board (~15W)
        self.R_CAMERAS = 73.0         # Cameras (~3W)
        self.R_ALFA = 109.5           # Alfa Network Card (~2W)
        
        # Motor Constants based on 2203.5 1500KV
        self.P_IDLE_SYSTEM = 4.0 * (10.0 * 0.3)  # 12W combined system idle power
        self.P_MAX_MOTOR_AERO = 288.0            # Max combined aerodynamic prop power at full throttle
        
        self.last_time = self.get_clock().now()

        # Telemetry log storage list
        self.telemetry_log = []
        
        # 2. ROS 2 SUBSCRIBER
        # Listens directly to PX4 actuator outputs
        self.subscription = self.create_subscription(
            ActuatorOutputs,
            '/fmu/out/actuator_outputs',
            self.actuator_callback,
            10)
        
        self.get_logger().info("Starling 2 Max Thévenin Battery Model Node Initialized.")

    def get_cell_v_oc(self, soc):
        """ Polynomial 4th-order curve-fit for a single Sony VTC6 cell """
        soc = max(0.0, min(soc, 1.0))
        v_cell = (4.15 - 0.4 * (1 - soc) - 0.3 * (1 - soc)**2 
                  + 0.5 * (1 - soc)**3 - 0.8 * (1 - soc)**4 + 0.15 * soc)
        return max(3.0, min(v_cell, 4.2))

    def actuator_callback(self, msg):
        # Calculate time delta for exact Coulomb counting
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time
        if dt <= 0:
            return

        # 1. Update State of Charge (SoC) and Open Circuit Voltage (V_th)
        soc = self.remaining_capacity_as / self.total_capacity_as
        v_th = self.get_cell_v_oc(soc) * self.S_COUNT
        
        # 2. Parse Motor Control Commands from PX4
        motor_commands = msg.output[:4]
        avg_throttle = sum(motor_commands) / 4.0
        
        # Handle scaling if PX4 sends raw PWM (e.g., 1000 to 2000) instead of normalized 0-1 float
        if avg_throttle > 1.0:
            avg_throttle = (avg_throttle - 1000.0) / 1000.0
            avg_throttle = max(0.0, min(avg_throttle, 1.0))

        # 3. Calculate Dynamic Aerodynamic Power based on true throttle curve
        p_propellers_dynamic = (avg_throttle ** 3) * self.P_MAX_MOTOR_AERO
        p_motors_total = p_propellers_dynamic + self.P_IDLE_SYSTEM
        
        # Avoid division by zero if completely powered off
        r_motors = (v_th ** 2) / p_motors_total if p_motors_total > 0 else 1e6
        
        # 4. Compute Combined Parallel Hardware Load Resistance
        inv_r_total_load = (1.0 / r_motors) + (1.0 / self.R_BOARD) + (1.0 / self.R_CAMERAS) + (1.0 / self.R_ALFA)
        r_total_load = 1.0 / inv_r_total_load
        
        # 5. Solve the Thévenin Matrix
        i_total_load = v_th / (self.R_TH + r_total_load)
        v_esc = v_th - (i_total_load * self.R_TH)
        
        # Calculate individual component powers for tracking logs
        p_board = (v_esc ** 2) / self.R_BOARD
        p_cameras = (v_esc ** 2) / self.R_CAMERAS
        p_alfa = (v_esc ** 2) / self.R_ALFA
        
        # 6. Deplete Capacity (Coulomb Counting)
        self.remaining_capacity_as -= i_total_load * dt
        if self.remaining_capacity_as < 0:
            self.remaining_capacity_as = 0.0

        # 7. Append data point to the telemetry log array
        self.telemetry_log.append([
            current_time.nanoseconds / 1e9,  # Timestamp
            v_th,                            # Chemical Volts
            v_esc,                           # Sagged Volts at ESC
            i_total_load,                    # System Amps
            p_motors_total,                  # Power consumed by 2203.5 Motors
            p_board + p_cameras + p_alfa     # Companion Computer Payload Power (Fixed ~20W)
        ])
            
        self.get_logger().info(
            f"SoC: {soc*100:.1f}% | V_Rail(Sagged): {v_esc:.2f}V | Current: {i_total_load:.2f}A"
        )

    def save_log_to_file(self, filename="run_power_dynamics.csv"):
        """ Exports the collected telemetry to a CSV spreadsheet """
        self.get_logger().info(f"Writing {len(self.telemetry_log)} data points to {filename}...")
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "V_th", "V_esc", "I_load", "P_motors", "P_payload"])
            writer.writerows(self.telemetry_log)
        self.get_logger().info("CSV Export Complete successfully.")

    def destroy_node(self):
        """ Triggers automatically when you terminate the node with Ctrl+C """
        self.save_log_to_file()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DroneTheveninBatteryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
