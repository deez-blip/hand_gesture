#!/usr/bin/env python3
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SocketIOClient(Node):
    """
    Bridge minimal:
    - self-test: 'droite' ~1s au démarrage puis 'stop'
    - souscrit à 'gesture_cmd' (run_realtime) et mappe vers on_commandeMotor(...)
    - publie sur 'driver_topic' un JSON strict: { "motorControl": { "motor1Speed": X, "motor2Speed": Y }, "screenControl": {...} }
    """
    def __init__(self):
        super().__init__('socketio_client')
        self.get_logger().info('Mode TEST: Ecoute de gesture_cmd + self-test 1s.')

        self.publisher = self.create_publisher(String, 'driver_topic', 1)

        # Souscription au topic de gestes
        self.gesture_sub = self.create_subscription(
            String,
            'gesture_cmd',
            self._gesture_cb,
            10
        )

        # Self-test (une seule fois): 'droite' ~1s puis 'stop'
        self.selftest_done = False
        self.selftest_timer = self.create_timer(1.0, self._selftest_timer_cb)
        self.stop_timer = None

    # --------- Self-test ----------
    def _selftest_timer_cb(self):
        """Active les moteurs à droite pendant ~1s, puis stop."""
        if self.selftest_done:
            try:
                self.selftest_timer.cancel()
            except Exception:
                pass
            return

        self.get_logger().info("Self-test: 'droite' (1s)")
        try:
            self.on_commandeMotor("droite")
        finally:
            self.selftest_done = True
            self.stop_timer = self.create_timer(1.0, self._selftest_stop_once)
            try:
                self.selftest_timer.cancel()
            except Exception:
                pass

    def _selftest_stop_once(self):
        """Stop unique après le self-test."""
        try:
            self.get_logger().info("Self-test: stop")
            self.on_commandeMotor("stop")
        finally:
            if self.stop_timer is not None:
                try:
                    self.stop_timer.cancel()
                except Exception:
                    pass
                self.stop_timer = None

    # --------- Gestes ----------
    def _gesture_cb(self, msg: String):
        if not self.selftest_done:
            self.selftest_done = True
            try:
                self.selftest_timer.cancel()
            except Exception:
                pass
            if self.stop_timer is not None:
                try:
                    self.stop_timer.cancel()
                except Exception:
                    pass
                self.stop_timer = None

        label = (msg.data or '').strip().lower()
        mapping = {
            'stop': 'stop',
            'avance': 'haut',
            'recul': 'bas',
            'droite': 'droite',
            'gauche': 'gauche',
        }
        cmd = mapping.get(label)
        if cmd:
            self.get_logger().info(f"[gesture_cmd] {label} -> on_commandeMotor('{cmd}')")
            self.on_commandeMotor(cmd)
        else:
            self.get_logger().warn(f"[gesture_cmd] label inconnu: '{label}'")

    # --------- Construction JSON strict + publication ----------
    def on_commandeMotor(self, data: str):
        """Mappe la commande symbolique vers vitesses et publie JSON strict accepté par ton .ino."""
        self.get_logger().info(f"Commande moteur : {data}")
        motor1, motor2 = 0, 0
        if data == "haut":
            motor1, motor2 = 250, 250
        elif data == "bas":
            motor1, motor2 = -250, -250
        elif data == "gauche":
            motor1, motor2 = 250, -250
        elif data == "droite":
            motor1, motor2 = -250, 250
        elif data == "stop":
            motor1, motor2 = 0, 0

        try:
            motor_json = {
                "motorControl": {
                    "motor1Speed": motor1,
                    "motor2Speed": motor2
                },
                "screenControl": {
                    "txt": f"Vit : M1={motor1}, M2={motor2}"
                }
            }
            msg = String()
            msg.data = json.dumps(motor_json)   # <— EXACTEMENT ce que ton .ino lit
            self.publisher.publish(msg)
            self.get_logger().info(f"JSON publié sur driver_topic : {msg.data}")
        except Exception as e:
            self.get_logger().error(f"Erreur lors de la publication du JSON : {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SocketIOClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()