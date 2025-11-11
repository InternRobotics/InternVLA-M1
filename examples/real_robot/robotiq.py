from __future__ import annotations

import os
import pyRobotiqGripper
from pymodbus.client.sync import ModbusSerialClient as ModbusClient


try:
    gripper = pyRobotiqGripper.RobotiqGripper()
    gripper.activate()
except Exception as e:
    print(f"Failed to connect to Robotiq gripper: {e}")
    gripper = None

class RobotiqCGripper(object):
    """Robotiq 2F gripper controller via Modbus RTU over serial."""

    def __init__(self, port='/dev/ttyUSB0'):
        os.system("dmesg|grep ttyUSB*")
        self.gripper_speed = 0.8
        self.gripper_force = 0.2
        self.client = ModbusClient(method='rtu', port=port, stopbits=1, bytesize=8, baudrate=115200, timeout=0.2)
    
    def wait_for_connection(self):
        """Wait for connection to gripper."""
        self.client.connect()

    def _get_bit(self, number, bit) -> bool:
        """Get specific bit value from a number."""
        return (number >> bit) & 1
    
    def _read_and_check_input_register(self, address) -> int:
        """Read input register from gripper."""
        feedback = self.client.read_input_registers(address=address, count=1, unit=0x0009)
        register_value = feedback.registers[0]
        return register_value

    def set_gripper_position(self, position):
        """Set gripper position (0.0 = closed, 1.0 = open)."""
        position_value = int(position * 255)
        speed_value = int(self.gripper_speed * 255)
        force_value = int(self.gripper_force * 255)
        bytes = [0b00001001, 0, 0, position_value, speed_value, force_value]
        values = []
        if len(bytes) % 2 != 0:
            bytes.append(0)
        for i in range(0, int(len(bytes) / 2)):
            values.append((bytes[2 * i] << 8) + bytes[2 * i + 1])
        self.client.write_registers(0x03E8, values=values, unit=0x0009)
    
    def is_object_grasped(self) -> bool:
        """Check if object is grasped."""
        movement_register = self._read_and_check_input_register(address=0x07D0)
        bit_15 = self._get_bit(movement_register, 15)
        bit_14 = self._get_bit(movement_register, 14)

        if (bit_15 == 1 and bit_14 == 0):
            return 1.0
        else:
            return 0.0
    
    def open(self):
        self.set_gripper_position(position=0.1)

    def close(self):
        self.set_gripper_position(position=0.9)

    def get_current_width(self):
        return self._read_and_check_input_register(address=0x07D4) / 255.0

    