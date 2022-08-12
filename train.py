# Solver for Dynamic VRPTW, baseline strategy is to use the static solver HGS-VRPTW repeatedly
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import argparse
import sys
import uuid
import torch
import tools
from environment import VRPEnvironment
from baselines.strategies import _filter_instance
from learning_method.nets.encoders.gnn_encoder import GNNEncoder
from learning_method.nets.encoders.our_model import AttentionModel
from solver import solve_static_vrptw
import threading

learning_rate = 1e-5


def episode_eval(model_name, args, instance):
    model = AttentionModel(encoder_class={"gnn": GNNEncoder}.get(args.encoder),
                           embedding_dim=args.embedding_dim,
                           n_encode_layers=args.n_encode_layers,
                           aggregation=args.aggregation,
                           normalization=args.normalization,
                           learn_norm=args.learn_norm,
                           track_norm=args.track_norm,
                           gated=args.gated)
    state_dict = torch.load("./models/" + model_name)
    model.load_state_dict(state_dict)
    model.eval()
    env = VRPEnvironment(instance=tools.read_vrplib("./dataset/test/" + instance),
                         epoch_tlim=60, is_static=False)
    total_reward = 0
    done = False
    observation, static_info = env.reset()
    epoch_tlim = static_info['epoch_tlim']
    while not done:
        epoch_instance = observation['epoch_instance']
        if len(epoch_instance['request_idx']) - 1 > 0:
            nodes_feature = tools.get_epoch_nodes_feature(epoch_instance, static_info)
            nodes = torch.tensor(nodes_feature)
            graph = torch.ones((len(epoch_instance['request_idx']), len(epoch_instance['request_idx'])))
            for idx in range(len(epoch_instance['request_idx'])):
                graph[idx][idx] = 0
            nodes = torch.unsqueeze(nodes, 0).to(torch.float32)
            graph = torch.unsqueeze(graph, 0).to(torch.long)
            pred_val, nodes_prob = model(nodes, graph)
        else:
            nodes_prob = []
        assignments_results = tools.get_assignment_results(nodes_prob, epoch_instance['must_dispatch'],
                                                           args.eval_policy, args)
        epoch_instance_dispatch = _filter_instance(epoch_instance, assignments_results)
        tmp_dir = os.path.join("tmp", str(uuid.uuid4()))
        epoch_solution, epoch_cost = list(solve_static_vrptw(epoch_instance_dispatch, time_limit=epoch_tlim,
                                                             tmp_dir=tmp_dir))[-1]
        tools.cleanup_tmp_dir(tmp_dir)
        # Map solution to indices of corresponding requests
        epoch_solution = [epoch_instance_dispatch['request_idx'][route] for route in epoch_solution]
        # step to next state
        observation, reward, done, info = env.step(epoch_solution)
        # assert epoch_cost is None or reward == -epoch_cost, "Reward should be negative cost of solution"
        assert not info['error'], f"Environment error: {info['error']}"

        total_reward += reward
    file = open("results/obj_results_raw.txt", "a")
    file.write(model_name + "," + str(sum(env.final_costs.values())) + "\n")
    file.close()


def eval_on_test_set(model_name, args):
    test_instances = os.listdir("./dataset/test/")
    for instance in test_instances:
        print("evaluating " + model_name + " on " + instance + "...")
        thread_pool = []
        for no_episode in range(5):
            t = threading.Thread(target=episode_eval, args=(model_name, args, instance))
            thread_pool.append(t)
            t.start()
    time.sleep(600)


def episode_train(model, args, training_instances, loss_func, optimizer):
    pred_batch = []
    epoch_reward = []
    total_reward = 0
    done = False
    env = VRPEnvironment(seed=episode_no,
                         instance=tools.read_vrplib(
                             "./dataset/multi_training/" + random.choice(training_instances)),
                         epoch_tlim=args.epoch_tlim, is_static=False)
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
            pred_batch.append(pred_val)
        else:
            nodes_prob = []
        assignments_results = tools.get_assignment_results(nodes_prob, epoch_instance['must_dispatch'],
                                                           args.train_policy, args)
        epoch_instance_dispatch = _filter_instance(epoch_instance, assignments_results)
        sol_cost_list = list(solve_static_vrptw(epoch_instance_dispatch, time_limit=epoch_tlim, tmp_dir=args.tmp_dir))
        if len(sol_cost_list) == 0:
            print("heuristic solver failed!")
            return
        epoch_solution, epoch_cost = sol_cost_list[-1]
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
        epoch_reward.append(reward)
        total_reward += reward
    if args.verbose:
        log(f"Cost of solution: {-total_reward}")
    if args.new_reward:
        reward_tensor = torch.tensor(tools.get_accumulated_reward_gap(epoch_reward))
    else:
        reward_tensor = torch.ones_like(torch.cat(pred_batch))
        reward_tensor[:, :] = float(total_reward)
    loss = loss_func(torch.cat(pred_batch), reward_tensor)
    loss.backward()
    optimizer.step()


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
    parser.add_argument("--max_episodes", type=int, default=120, help="maximum training episodes")

    # model
    parser.add_argument('--encoder', default="gnn",
                        help='Type of encoder')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=3,
                        help='Number of layers in the encoder/critic network')
    parser.add_argument('--aggregation', default='mean',
                        help="Neighborhood aggregation function: 'sum'/'mean'/'max'")
    parser.add_argument('--aggregation_graph', default='mean',
                        help="Graph embedding aggregation function: 'sum'/'mean'/'max'")
    parser.add_argument('--normalization', default='layer',
                        help="Normalization type: 'batch'/'layer'/None")
    parser.add_argument('--learn_norm', action='store_true',
                        help="Enable learnable affine transformation during normalization")
    parser.add_argument('--track_norm', action='store_true',
                        help="Enable tracking batch statistics during normalization")
    parser.add_argument('--gated', action='store_true',
                        help="Enable edge gating during neighborhood aggregation")
    parser.add_argument('--n_heads', type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument('--alias', default='',
                        help="Denote the model trained")
    parser.add_argument('--eval_policy', default='greedy',
                        help="Decode policy for evaluation")
    parser.add_argument('--train_policy', default='greedy',
                        help="Decode policy for training")
    parser.add_argument('--greedy_threshold', type=float, default=1.0,
                        help="Threshold for greedy policy")
    parser.add_argument('--new_reward', action='store_true',
                        help="Threshold for greedy policy")
    parser.add_argument('--model_name', type=str, default=None,
                        help="Threshold for greedy policy")

    args = parser.parse_args()
    training_config = args.encoder + "_" + str(args.embedding_dim) + "_" + str(
        args.n_encode_layers) + "_" + args.aggregation + "_" + args.train_policy + "_" + args.eval_policy + "_" + str(
        args.greedy_threshold) + "_" + args.alias + "_"
    if args.tmp_dir is None:
        # Generate random tmp directory
        args.tmp_dir = os.path.join("tmp", str(uuid.uuid4()))
        cleanup_tmp_dir = True
    else:
        # If tmp dir is manually provided, don't clean it up (for debugging)
        cleanup_tmp_dir = False
    training_instances = os.listdir("./dataset/multi_training")
    model = AttentionModel(encoder_class={"gnn": GNNEncoder}.get(args.encoder),
                           embedding_dim=args.embedding_dim,
                           n_encode_layers=args.n_encode_layers,
                           aggregation=args.aggregation,
                           normalization=args.normalization,
                           learn_norm=args.learn_norm,
                           track_norm=args.track_norm,
                           gated=args.gated)
    if args.model_name is not None:
        print("continue to train ", args.model_name)
        state_dict = torch.load("./models/" + args.model_name)
        model.load_state_dict(state_dict)
    model.train()
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    try:
        for episode_no in range(args.max_episodes):
            optimizer.zero_grad()
            episode_train(model, args, training_instances, loss_func, optimizer)
            if (episode_no + 1) % 100 == 0:
                torch.save(model.state_dict(), "./models/" + training_config + str(episode_no))
                eval_on_test_set(training_config + str(episode_no), args)
    finally:
        if cleanup_tmp_dir:
            tools.cleanup_tmp_dir(args.tmp_dir)
