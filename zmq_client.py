import zmq
import argparse
from benchmark_server.ideal_world_bridge import ideal_world_interface
from benchmark_server.ideal_world_bridge.case_definition_pb2 import BridgeReplyMessage, BridgeRequestMessage

parser = argparse.ArgumentParser("zmq_client")
parser.add_argument(
    "--addr", type=str, default=None,
    help='The addr which will be connected to. It should be zmq addr form.'
)

args = parser.parse_args()

context = zmq.Context()

reply = BridgeReplyMessage()
request = BridgeRequestMessage()

request.state = ideal_world_interface.KNOCK_MSG

#  Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)
socket.connect(args.addr)

socket.send(request.SerializeToString())
message = socket.recv()
reply.ParseFromString(message)
print("REP: ", reply)
if reply.state != ideal_world_interface.REP_KNOCK_MSG:
    exit()

for i in range(10):
    del request.traj.traj_points[:]
    traj = request.traj
    traj_point = traj.traj_points.add()
    traj_point.x = 0
    traj_point.y = 0
    traj_point.theta = 0
    traj_point.vel = 1

    traj_point = traj.traj_points.add()
    traj_point.x = 0
    traj_point.y = 1
    traj_point.theta = 0
    traj_point.vel = 1

    traj_point = traj.traj_points.add()
    traj_point.x = 0
    traj_point.y = 2
    traj_point.theta = 0
    traj_point.vel = 1

    traj_point = traj.traj_points.add()
    traj_point.x = 0
    traj_point.y = 3
    traj_point.theta = 0
    traj_point.vel = 1

    message = request.SerializeToString()

    print("-----------Send [%s]----------" % i)
    print(request)
    socket.send(message)

    #  Get the reply.
    message = socket.recv()
    reply.ParseFromString(message)
    print("-----------Reply [%s]----------" % i)
    print("Received reply: ", reply)