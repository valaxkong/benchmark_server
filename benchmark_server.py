# Copyright (C) 2020. UISEE Technologies Co., Ltd. All rights reserved.
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
import time
import logging
import argparse
import ray
from pathlib import Path

from typing import Sequence, Dict
from gym.spaces import Tuple

import smarts
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.scenario import Scenario
from smarts.core.utils.visdom_client import VisdomClient
from envision.client import Client as Envision

from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent_manager import AgentManager

from smarts.benchmark.metrics.basic_metrics import Metric

from examples import default_argument_parser

from ideal_world_bridge.ideal_world_interface import RemoteIdealWorldEgo
from smarts.core.controllers import ActionSpaceType

import socket

'''Currently, Benchmark Server just support one scenario one client fot testing.'''

def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip

def get_addr_str_for_zmq():

    host_ip = get_host_ip()
    return "tcp://" + host_ip + ":8078"

print("Benchmark Server will be launch at: ", get_addr_str_for_zmq())


EGO_ID = "Agent-007"
RUN_NAME = Path(__file__).stem
EXPERIMENT_NAME = "{scenario}-{n_agent}"

class BenchmarkServer:
    '''A complete remote benchmark environment that warps a SMARTS simulation.

        scenarios:
            a list of directories of the scenarios that will be run
        agent_specs:
            a dict of agentspecs that will run in the environment
        headless:
            true|false envision disabled
        visdom:
            true|false visdom integration
        timestep_sec:
            the step length for all components of the simulation
        seed:
            the seed for random number generation
        num_external_sumo_clients:
            the number of SUMO clients beyond SMARTS
        sumo_headless:
            true|false for SUMO visualization disabled [sumo-gui|sumo]
        sumo_port:
            used to specify a specific sumo port
        sumo_auto_start:
            true|false sumo will start automatically
        envision_endpoint:
            used to specify envision's uri
        envision_record_data_replay_path:
            used to specify envision's data replay output directory
        zoo_workers:
            List of (ip, port) tuples of Zoo Workers, used to instantiate remote social agents
        auth_key:
            Authentication key of type string for communication with Zoo Workers
    '''

    def __init__(
        self,
        scenarios: Sequence[str],
        agent_specs: Dict,
        shuffle_scenarios=True,
        headless=False,
        visdom=False,
        timestep_sec=0.1,
        seed=42,
        num_external_sumo_clients=0,
        sumo_headless=True,
        sumo_port=None,
        sumo_auto_start=True,
        endless_traffic=True,
        envision_endpoint=None,
        envision_record_data_replay_path=None,
        zoo_workers=None,
        auth_key=None,
    ):
        self._metircs = Metric(1)

        self.has_connection = False

        self._log = logging.getLogger(self.__class__.__name__)
        smarts.core.seed(seed) # Set seed for np and random module.

        self._agent_specs = agent_specs
        self._dones_registered = 0

        # Setup ego.
        self._ego = agent_specs[EGO_ID].build_agent()

        # Setup sceanrios for benchmark.
        self._scenarios_iterator = Scenario.scenario_variations(
            scenarios, list(agent_specs.keys()), shuffle_scenarios,
        )

        # Setup envision and visdom.
        envision_client = None
        if not headless:
            envision_client = Envision(
                endpoint=envision_endpoint, output_dir=envision_record_data_replay_path
            )

        visdom_client = None
        if visdom:
            visdom_client = VisdomClient()

        # Setup SMARTS
        agent_interfaces = {
            agent_id: agent.interface for agent_id, agent in agent_specs.items()
        }

        self._smarts = SMARTS(
            agent_interfaces=agent_interfaces,
            traffic_sim=SumoTrafficSimulation(
                headless=sumo_headless,
                time_resolution=timestep_sec,
                num_external_sumo_clients=num_external_sumo_clients,
                sumo_port=sumo_port,
                auto_start=sumo_auto_start,
                endless_traffic=endless_traffic,
            ),
            envision=envision_client,
            visdom=visdom_client,
            timestep_sec=timestep_sec,
            zoo_workers=zoo_workers,
            auth_key=auth_key,
        )

    @property
    def scenario_log(self):
        """Simulation step logs.

        Returns:
            A dictionary with the following:
                timestep_sec:
                    The timestep of the simulation.
                scenario_map:
                    The name of the current scenario.
                scenario_routes:
                    The routes in the map.
                mission_hash:
                    The hash identifier for the current scenario.
        """

        scenario = self._smarts.scenario
        return {
            "timestep_sec": self._smarts.timestep_sec,
            "scenario_map": scenario.name,
            "scenario_routes": scenario.route or "",
            "mission_hash": str(hash(frozenset(scenario.missions.items()))),
        }

    def run_benchmark(self):
        '''Main processdure of benchmarking.'''

        proto_obs = dict()
        try:
            proto_obs = self.reset()
        except StopIteration:
            print("No sceanrio has been setted for testing!")
            return

        while True:
            if self.has_connection:
                try:
                    proto_obs = self.reset()
                except StopIteration:
                    print("All specified scenarios has been tested!")
                    break
            else:
                self._ego.block_for_connection(proto_obs[EGO_ID])
                self.has_connection = True

            dones = {"__all__": False}
            while not dones["__all__"]:
                proto_ego_obs = proto_obs[EGO_ID]
                proto_ego_act = self._ego.act(proto_ego_obs, self._is_reset)
                proto_obs, _, dones, _ = self.step({EGO_ID: proto_ego_act})
            metric_res = self._metircs.compute()
            print("++++Metric++++\n", metric_res)

    def step(self, agent_actions):
        '''Input serilized action and out serilized observation as well'''

        agent_actions = {
            agent_id: self._agent_specs[agent_id].action_adapter(action)
            for agent_id, action in agent_actions.items()
        }

        observations, rewards, agent_dones, infos = self._smarts.step(agent_actions)
        self._metircs.log_step(observations, rewards, agent_dones, infos, 0)
        print("smarts ego action : ", agent_actions[EGO_ID])
        print("smarts ego observation: ", observations[EGO_ID])

        obs = dict()
        for agent_id in observations:
            agent_spec = self._agent_specs[agent_id]
            observation = observations[agent_id]
            obs[agent_id] = agent_spec.observation_adapter(observation)

        for done in agent_dones.values():
            self._dones_registered += 1 if done else 0

        agent_dones["__all__"] = self._dones_registered == len(self._agent_specs)

        self._is_reset = False
        return obs, _, agent_dones, _

    def reset(self) -> Dict:
        self._metircs.reset()

        scenario = next(self._scenarios_iterator)

        self._dones_registered = 0
        env_observations = self._smarts.reset(scenario)

        obs = {
            agent_id: self._agent_specs[agent_id].observation_adapter(observation)
            for agent_id, observation in env_observations.items()
        }

        self._is_reset = True
        obs[EGO_ID].mov_objs[0].pose2d.pos.x = 30.0
        print('++++++++++++++RESET+++++++++++')
        return obs

    def render(self, mode="human"):
        """Does nothing."""
        pass

    def close(self):
        if self._smarts is not None:
            self._smarts.destroy()

def main(scenarios, headless, seed, auth_key=None):

    # Setup AgentSpec for tested agent
    ego_spec = AgentSpec(
        interface=AgentInterface.from_type(AgentType.TrajectoryInterpolator, neighborhood_vehicles=True),
        action_adapter=RemoteIdealWorldEgo.action_adapter,
        observation_adapter=RemoteIdealWorldEgo.observation_adapter,
        agent_builder=RemoteIdealWorldEgo,
        agent_params=get_addr_str_for_zmq()
    )

    # Setup Benchmark
    benchmark_server = BenchmarkServer(
        scenarios=scenarios,
        agent_specs={EGO_ID: ego_spec},
        headless=headless,
        timestep_sec=0.1,
        seed=seed,
        auth_key=auth_key
    )

    benchmark_server.run_benchmark()

def parse_args():
    parser = argparse.ArgumentParser("Benchmark learning")
    parser.add_argument(
        "--scenarios", type=str, help="a list of directories of the scenarios that will be run",
    )
    # parser.add_argument(
    #     "--paradigm",
    #     type=str,
    #     default="decentralized",
    #     help="Algorithm paradigm, decentralized (default) or centralized",
    # )
    parser.add_argument(
        "--headless", default=True, action="store_true", help="Turn on headless mode"
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="Seed for scenario generation"
    )
    # parser.add_argument(
    #     "--log_dir",
    #     default="./log/results",
    #     type=str,
    #     help="Path to store RLlib log and checkpoints, default is ./log/results",
    # )
    # parser.add_argument("--config_file", "-f", type=str, required=True)
    # parser.add_argument("--restore_path", type=str, default=None)
    # parser.add_argument("--num_workers", type=int, default=1, help="RLlib num workers")
    # parser.add_argument("--cluster", action="store_true")
    # parser.add_argument(
    #     "--num_episodes", type=int, default=1000, help="num of episode"
    # )

    return parser.parse_args()

if __name__ == "__main__":
    parser = default_argument_parser("Benchmark server")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        seed=args.seed
    )