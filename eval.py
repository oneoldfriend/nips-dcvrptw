# Solver for Dynamic VRPTW, baseline strategy is to use the static solver HGS-VRPTW repeatedly
import os
import random
import argparse
import math
import sys
import uuid
import torch
import tools
from environment import VRPEnvironment
from learning_method.nets.encoders.gnn_encoder import GNNEncoder
from learning_method.nets.encoders.our_model import AttentionModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def run_on_instance(model, env):
    total_reward = 0
    done = False
    observation, static_info = env.reset()
    epoch_tlim = static_info['epoch_tlim']
    num_requests_postponed = 0
    while not done:
        epoch_instance = observation['epoch_instance']
        if len(epoch_instance['request_idx']) - 1 > 0:
            nodes_feature = tools.get_epoch_nodes_feature(epoch_instance, static_info)
            nodes = torch.tensor(nodes_feature)
            graph = torch.ones((len(epoch_instance['request_idx']), len(epoch_instance['request_idx'])))
            for idx in range(len(epoch_instance['request_idx'])):
                graph[idx][idx] = 0
            if args.verbose:
                log(f"Epoch {static_info['start_epoch']} <= {observation['current_epoch']} <= {static_info['end_epoch']}",
                    newline=False)
                num_requests_open = len(epoch_instance['request_idx']) - 1
                num_new_requests = num_requests_open - num_requests_postponed
                log(f" | Requests: +{num_new_requests:3d} = {num_requests_open:3d}, {epoch_instance['must_dispatch'].sum():3d}/{num_requests_open:3d} must-go...",
                    newline=False, flush=True)
            nodes = torch.unsqueeze(nodes, 0).to(torch.float32)
            graph = torch.unsqueeze(graph, 0).to(torch.long)
            pred_val, nodes_prob = model(nodes, graph)
        else:
            nodes_prob = []
        assignments_results = tools.get_assignment_results(nodes_prob, epoch_instance['must_dispatch'])
        epoch_instance_dispatch = _filter_instance(epoch_instance, assignments_results)
        epoch_solution, epoch_cost = list(
            solve_static_vrptw(epoch_instance_dispatch, time_limit=math.ceil(epoch_tlim * 1 / 2), tmp_dir=args.tmp_dir,
                               seed=args.solver_seed))[-1]
        # Map solution to indices of corresponding requests
        epoch_solution = [epoch_instance_dispatch['request_idx'][route] for route in epoch_solution]

        if args.verbose:
            num_requests_dispatched = sum([len(route) for route in epoch_solution])
            num_requests_open = len(epoch_instance['request_idx']) - 1
            num_requests_postponed = num_requests_open - num_requests_dispatched
            log(f" {num_requests_dispatched:3d}/{num_requests_open:3d} dispatched and {num_requests_postponed:3d}/{num_requests_open:3d} postponed | Routes: {len(epoch_solution):2d} with cost {epoch_cost:6d}")

        # step to next state
        observation, reward, done, info = env.step(epoch_solution)
        assert epoch_cost is None or reward == -epoch_cost, "Reward should be negative cost of solution"
        assert not info['error'], f"Environment error: {info['error']}"

        total_reward += reward
    if args.verbose:
        log(f"Cost of solution: {-total_reward}")
    return total_reward


def log(obj, newline=True, flush=False):
    # Write logs to stderr since program uses stdout to communicate with controller
    sys.stderr.write(str(obj))
    if newline:
        sys.stderr.write('\n')
    if flush:
        sys.stderr.flush()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch_tlim", type=int, default=120, help="Time limit per epoch")
    parser.add_argument("--tmp_dir", type=str, default=None,
                        help="Provide a specific directory to use as tmp directory (useful for debugging)")
    parser.add_argument("--verbose", action='store_true', help="Show verbose output")
    parser.add_argument("--running_times", type=int, default=5, help="maximum running times per instances")

    args = parser.parse_args()
    training_config = args.encoder + str(args.embedding_dim) + str(
        args.n_encode_layers) + args.aggregation + args.normalization + args.training_episodes
    if args.tmp_dir is None:
        # Generate random tmp directory
        args.tmp_dir = os.path.join("tmp", str(uuid.uuid4()))
        cleanup_tmp_dir = True
    else:
        # If tmp dir is manually provided, don't clean it up (for debugging)
        cleanup_tmp_dir = False
    test_instances = os.listdir("./dataset/test/")
    model = AttentionModel(embedding_dim=args.embedding_dim,
                           encoder_class={"gnn": GNNEncoder}.get(args.encoder),
                           n_encode_layers=args.n_encode_layers,
                           aggregation=args.aggregation,
                           normalization=args.normalization,
                           learn_norm=args.learn_norm,
                           track_norm=args.track_norm,
                           gated=args.gated)
    model.eval()
    try:
        for episode_no in range(args.running_times):
            env = VRPEnvironment(seed=args.instance_seed,
                                 instance=tools.read_vrplib("./dataset/training/" + random.choice(test_instances)),
                                 epoch_tlim=args.epoch_tlim, is_static=False)
            episodes_train(model, env)
            if episode_no % 100 == 0:
                torch.save(model.state_dict(), "./models/" + training_config + ".pth")

    finally:
        if cleanup_tmp_dir:
            tools.cleanup_tmp_dir(args.tmp_dir)
