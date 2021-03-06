# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import binascii
import numpy as np
from . import case_definition_pb2
import zmq
import time
import math
from smarts.core.coordinates import Heading

KNOCK_MSG = 'This is IdealWorld'
REP_KNOCK_MSG = 'This is Benchmark'
HALF_PI = math.pi / 2
PI = math.pi

class RemoteIdealWorldEgo:

    def __init__(self, server_addr: str):
        # Setup network related
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind(server_addr)

    def block_for_connection(self, proto_obs):
        # Waiting for connection, reply with REP_KNOCK_MSG

        proto_rep = case_definition_pb2.BridgeReplyMessage()
        proto_rep.observation.CopyFrom(proto_obs) ## TODO: non-copy implementation

        while True:
            print("Waiting for connection...")
            message = self._socket.recv() ## Sync and block
            print("Received connection message: %s" % message)

            proto_req = case_definition_pb2.BridgeRequestMessage()
            proto_req.ParseFromString(message)
            print("Received connection request: %s" % proto_req.state)

            if proto_req.state == KNOCK_MSG:
                proto_rep.state = REP_KNOCK_MSG
                print("init proto rep: ", proto_rep)
                self._socket.send(proto_rep.SerializeToString())
                print("IdealWorld client connected!")
                break
            else:
                proto_rep.state = 'Reject connection'
                self._socket.send(proto_rep.SerializeToString())
                print("Unknown connection request!")

    def generate_stop_proto_traj(proto_mov_obj):
        ''' Generate a stop traj according with a protobuf mov obj struct.
            The definition of 'stop traj' can be found at
            TrajectoryInterpolationController.perform_trajectory_interpolation().
        '''
        traj = case_definition_pb2.Trajectory()
        traj_point = traj.traj_points.add()

        traj_point.x = proto_mov_obj.pose2d.pos.x
        traj_point.y = proto_mov_obj.pose2d.pos.y
        traj_point.theta = proto_mov_obj.pose2d.theta
        traj_point.vel = proto_mov_obj.vel
        traj_point.t = 0.0

        return traj


    def send_observation(self, proto_ego_obs, is_reset):
        ''' Observation has been transferred by adapter'''

        proto_rep = case_definition_pb2.BridgeReplyMessage()
        proto_rep.observation.CopyFrom(proto_ego_obs)

        if is_reset:
            proto_rep.state = 'Reset'
        else:
            proto_rep.state = 'Observation'

        self._socket.send(proto_rep.SerializeToString())

    def recv_action(self):

        req_message = self._socket.recv()

        proto_req = case_definition_pb2.BridgeRequestMessage()
        proto_req.ParseFromString(req_message)

        return proto_req.traj

    @staticmethod
    def action_adapter(ideal_world_traj):
        '''Transfer proto trajectory to SMARTS trajectory'''
        # print("action adapter action traj: ", ideal_world_traj)

        t = []
        x = []
        y = []
        theta = []
        speed = []

        for traj_point in ideal_world_traj.traj_points:
            t.append(traj_point.t)
            x.append(traj_point.x)
            y.append(traj_point.y)
            theta.append(Heading(traj_point.theta - HALF_PI))
            speed.append(traj_point.vel)

        if len(x) < 2 or len(y) < 2 or len(theta) < 2 or len(speed) < 2:
            raise RuntimeError('Input traj lenth should be at least 2.')

        # To make sure the first point is ahead of ego.
        # TODO: Check the first point constrain here or let client-end to ensure this?
        traj = np.array([t[0:], x[0:], y[0:], theta[0:], speed[0:]])
        return traj

    @staticmethod
    def observation_adapter(observation):
        '''Transfer SMARTS observation to proto observation'''

        proto_obs = case_definition_pb2.Observation()

        ego = proto_obs.mov_objs.add()
        ego_vehicle_state = observation.ego_vehicle_state

        ego.description = 'vehicle.ego'
        ego.id = -1
        ego.geo.length = ego_vehicle_state.bounding_box.length
        ego.geo.width = ego_vehicle_state.bounding_box.width
        ego.vel = ego_vehicle_state.speed
        ego.pose2d.pos.x = ego_vehicle_state.position[0]
        ego.pose2d.pos.y = ego_vehicle_state.position[1]
        ego.pose2d.theta = ego_vehicle_state.heading + HALF_PI


        for i, vehicle_observation in enumerate(observation.neighborhood_vehicle_states):
            mov_obj = proto_obs.mov_objs.add()
            mov_obj.description = 'vehicle.agent'
            mov_obj.id = i ## TODO(kls): Tracking agent id between two frame

            mov_obj.geo.length = vehicle_observation.bounding_box.length
            mov_obj.geo.width = vehicle_observation.bounding_box.width

            mov_obj.vel = vehicle_observation.speed
            mov_obj.pose2d.pos.x = vehicle_observation.position[0]
            mov_obj.pose2d.pos.y = vehicle_observation.position[1]
            mov_obj.pose2d.theta = vehicle_observation.heading + HALF_PI

        # print("observation adapter smarts original observation:", ego_vehicle_state)
        # print("observation adapter observation:", proto_obs)
        return proto_obs
