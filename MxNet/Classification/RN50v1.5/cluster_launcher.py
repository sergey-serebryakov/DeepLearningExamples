# (c) Copyright [2017] Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Mxnet cluster launcher that is responsible for launching process on one node.

Mxnet cluster consists of three agents - one scheduler and multiple servers and
workers. This launcher runs a cluster with a specific topology:
1. There is only one scheduler running on some node.
2. Each node runs one worker and one server agents.
3. Each agent may use multiple GPUs.
4. All agents use the same number of GPUs (heterogeneous architecture).

Idea is similar to this one:
  https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py

The very fist agent that needs to run is the scheduler. Then all others can run.
"""
from __future__ import absolute_import
from __future__ import print_function
import argparse
import sys
import os
import subprocess
import time


def parse_args():
    """Parse command line arguments."""
    def str2bool(val):
        """Return true if val represents true  value."""
        return val.lower() in ('true', 'on', 't', '1')
    parser = argparse.ArgumentParser()

    parser.add_argument('--rendezvous', type=str, default='',
                        help='Scheduler rendezvous address IP:PORT.')

    parser.add_argument('--host-ip', type=str, default=None,
                        help='Network IP to use for this host.')
    parser.add_argument('--host-interface', type=str, default=None,
                        help='Network interface to use if host ip not provided.')

    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of worker processes.')
    parser.add_argument('--num_servers', type=int, default=-1,
                        help='Number of server processes. By default, equals to number of workers.')

    parser.add_argument('--scheduler', nargs='?', const=True, default=False, type=str2bool,
                        help='If a scheduler needs to be run on this node.')
    parser.add_argument('--server', nargs='?', const=True, default=True, type=str2bool,
                        help='If a server needs to be run on this node.')

    parser.add_argument('--ps-verbose', type=int, default=0)
    parser.add_argument("benchmark_script", type=str)
    parser.add_argument('benchmark_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.num_servers < 0:
        args.num_servers = args.num_workers
    elif args.num_servers == 0:
        args.server = False

    return args


def print_now(msg):
    """Print message and flush standard output."""
    print(msg)
    sys.stdout.flush()


class Cluster(object):
    """MXNET Cluster manager."""
    def __init__(self, args=None):
        if args is not None:
            self.num_workers = args.num_workers
            self.num_servers = args.num_servers
            self.is_scheduler = args.scheduler
            self.is_server = args.server
            self.host_ip = args.host_ip
            self.host_interface = args.host_interface
            self.ps_verbose_level = args.ps_verbose
            self.benchmark = {'script': args.benchmark_script, 'args': args.benchmark_args}

            scheduler = args.rendezvous.split(':')
            if len(scheduler) == 2:
                self.scheduler = {'host': scheduler[0], 'port': int(scheduler[1])}
            else:
                raise ValueError("Invalid rendezvous specifier format (%s)" % args.rendezvous)
        self.agents = {}

    def agent_specs(self, role):
        """Return agent specs for a particular role."""
        specs = {
            'DMLC_ROLE': role,
            'DMLC_PS_ROOT_URI': self.scheduler['host'], 'DMLC_PS_ROOT_PORT': self.scheduler['port'],
            'DMLC_NUM_SERVER': self.num_workers, 'DMLC_NUM_WORKER': self.num_servers,
            'PS_VERBOSE': self.ps_verbose_level
        }
        if self.host_ip is not None:
            specs['DMLC_NODE_HOST'] = self.host_ip
        elif self.host_interface is not None:
            specs['DMLC_INTERFACE'] = self.host_interface
        return specs

    def run_agent(self, role):
        """Run agent in a background process.
        Args:
            role: `str` An agent's role. One of 'scheduler' or 'server'.
        Returns:
            An instance of a subprocess.
        """
        env = os.environ.copy()
        cluster_vars = self.agent_specs(role)
        print_now("Running agent '%s' with cluster parameters '%s'" % (role, cluster_vars))
        env.update(cluster_vars)
        return subprocess.Popen(
            [sys.executable, '-u', '-c', 'import mxnet;'],
            shell=False,
            env=env
        )

    def launch(self):
        """Launch cluster."""
        if self.is_scheduler:
            self.agents['scheduler'] = self.run_agent('scheduler')
            time.sleep(5)
            if self.agents['scheduler'].poll() is not None:
                print_now("Scheduler was not started (return code=%s)" % (self.agents['scheduler'].poll()))
                exit(1)
        if self.is_server > 0:
            self.agents['server'] = self.run_agent('server')

        env = os.environ.copy()
        cluster_vars = self.agent_specs('worker')
        print_now("Running agent 'worker' with cluster parameters '%s'" % cluster_vars)
        env.update(cluster_vars)
        cmd = [sys.executable, '-u', self.benchmark['script']] + self.benchmark['args']
        self.agents['worker'] = subprocess.Popen(cmd, env=env)

    def wait(self):
        """Wait for a worker to complete."""
        print_now("Waiting for a worker.")
        self.agents['worker'].wait()

    def shutdown(self):
        """Shutdown.

        Generally, scheduler and server should shutdown themselves once all workers exited.
        """
        def _alive(role):
            agent = self.agents.get(role, None)
            return agent is not None and agent.poll() is None

        print_now("Shutting down agents.")
        for agent_role in ('server', 'scheduler'):
            if _alive(agent_role):
                self.agents[agent_role].terminate()


def main():
    """Runs agents on a local node."""
    cluster = Cluster(args=parse_args())
    cluster.launch()
    cluster.wait()
    cluster.shutdown()


if __name__ == "__main__":
    print_now("Running mxnet benchmarks with cluster launcher.")
    main()
