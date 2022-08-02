#!/usr/bin/env python
import os
import json
import pprint as pp
import numpy as np
import uuid
import sys

import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from options import get_options
from train import train_epoch, get_inner_model

from nets.attention_model import AttentionModel
from nets.nar_model import NARModel
from nets.critic_network import CriticNetwork
from nets.encoders.gat_encoder import GraphAttentionEncoder
from nets.encoders.gnn_encoder import GNNEncoder
from nets.encoders.mlp_encoder import MLPEncoder

from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline

from environment import VRPEnvironment, ControllerEnvironment

import tools
from utils import torch_load_cpu, load_problem

import warnings

warnings.filterwarnings("ignore",
                        message="indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead.")


def run(args):
    _run_rl(args)


def _run_rl(args):
    # Pretty print the run args
    pp.pprint(vars(args))

    # Set the random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not args.no_tensorboard:
        tb_logger = TbLogger(os.path.join(
            args.log_dir, "{}_{}-{}".format(args.problem, args.min_size, args.max_size), args.run_name))

    os.makedirs(args.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(args.save_dir, "args.json"), 'w') as f:
        json.dump(vars(args), f, indent=True)

    # Set the device
    args.device = torch.device("cuda:0" if args.use_cuda else "cpu")
    if args.tmp_dir is None:
        # Generate random tmp directory
        args.tmp_dir = os.path.join("tmp", str(uuid.uuid4()))
        cleanup_tmp_dir = True
    else:
        # If tmp dir is manually provided, don't clean it up (for debugging)
        cleanup_tmp_dir = False

    try:
        if args.instance is not None:
            env = VRPEnvironment(seed=args.instance_seed, instance=tools.read_vrplib(args.instance),
                                 epoch_tlim=args.epoch_tlim, is_static=args.static)
        else:
            assert args.strategy != "oracle", "Oracle can not run with external controller"
            # Run within external controller
            env = ControllerEnvironment(sys.stdin, sys.stdout)

        # Make sure these parameters are not used by your solver
        args.instance = None
        args.instance_seed = None
        args.static = None
        args.epoch_tlim = None

        # Figure out what's the problem
        problem = load_problem(args.problem)

        # Load data from load_path
        load_data = {}
        assert args.load_path is None or args.resume is None, "Only one of load path and resume can be given"
        load_path = args.load_path if args.load_path is not None else args.resume
        if load_path is not None:
            print('\nLoading data from {}'.format(load_path))
            load_data = torch_load_cpu(load_path)

        # Initialize model
        model_class = {
            'attention': AttentionModel,
            'nar': NARModel,
            # 'pointer': PointerNetwork
        }.get(args.model, None)
        assert model_class is not None, "Unknown model: {}".format(model_class)
        encoder_class = {
            'gnn': GNNEncoder,
            'gat': GraphAttentionEncoder,
            'mlp': MLPEncoder
        }.get(args.encoder, None)
        assert encoder_class is not None, "Unknown encoder: {}".format(encoder_class)
        model = model_class(
            problem=problem,
            embedding_dim=args.embedding_dim,
            encoder_class=encoder_class,
            n_encode_layers=args.n_encode_layers,
            aggregation=args.aggregation,
            aggregation_graph=args.aggregation_graph,
            normalization=args.normalization,
            learn_norm=args.learn_norm,
            track_norm=args.track_norm,
            gated=args.gated,
            n_heads=args.n_heads,
            tanh_clipping=args.tanh_clipping,
            mask_inner=True,
            mask_logits=True,
            mask_graph=False,
            checkpoint_encoder=args.checkpoint_encoder,
            shrink_size=args.shrink_size
        ).to(args.device)

        if args.use_cuda and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        # Compute number of network parameters
        print(model)
        nb_param = 0
        for param in model.parameters():
            nb_param += np.prod(list(param.data.size()))
        print('Number of parameters: ', nb_param)

        # Overwrite model parameters by parameters to load
        model_ = get_inner_model(model)
        model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

        # Initialize baseline
        if args.baseline == 'exponential':
            baseline = ExponentialBaseline(args.exp_beta)

        elif args.baseline == 'critic' or args.baseline == 'critic_lstm':
            baseline = CriticBaseline(
                (
                    CriticNetwork(
                        embedding_dim=args.embedding_dim,
                        encoder_class=encoder_class,
                        n_encode_layers=args.n_encode_layers,
                        aggregation=args.aggregation,
                        normalization=args.normalization,
                        learn_norm=args.learn_norm,
                        track_norm=args.track_norm,
                        gated=args.gated,
                        n_heads=args.n_heads
                    )
                ).to(args.device)
            )

            print(baseline.critic)
            nb_param = 0
            for param in baseline.get_learnable_parameters():
                nb_param += np.prod(list(param.data.size()))
            print('Number of parameters (BL): ', nb_param)

        elif args.baseline == 'rollout':
            baseline = RolloutBaseline(model, problem, args)

        else:
            assert args.baseline is None, "Unknown baseline: {}".format(args.baseline)
            baseline = NoBaseline()

        if args.bl_warmup_epochs > 0:
            baseline = WarmupBaseline(baseline, args.bl_warmup_epochs, warmup_exp_beta=args.exp_beta)

        # Load baseline from data, make sure script is called with same type of baseline
        if 'baseline' in load_data:
            baseline.load_state_dict(load_data['baseline'])

        # Initialize optimizer
        optimizer = optim.Adam(
            [{'params': model.parameters(), 'lr': args.lr_model}]
            + (
                [{'params': baseline.get_learnable_parameters(), 'lr': args.lr_critic}]
                if len(baseline.get_learnable_parameters()) > 0
                else []
            )
        )

        # Load optimizer state
        if 'optimizer' in load_data:
            optimizer.load_state_dict(load_data['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(args.device)

        # Initialize learning rate scheduler, decay by lr_decay once per epoch!
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: args.lr_decay ** epoch)

        # Load/generate datasets
        val_datasets = []
        for val_filename in args.val_datasets:
            val_datasets.append(
                problem.make_dataset(
                    filename=val_filename, batch_size=args.batch_size, num_samples=args.val_size,
                    neighbors=args.neighbors, knn_strat=args.knn_strat, supervised=True, nar=False
                ))

        if args.resume:
            epoch_resume = int(os.path.splitext(os.path.split(args.resume)[-1])[0].split("-")[1])

            torch.set_rng_state(load_data['rng_state'])
            if args.use_cuda:
                torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
            # Set the random states
            # Dumping of state was done before epoch callback, so do that now (model is loaded)
            baseline.epoch_callback(model, epoch_resume)
            print("Resuming after {}".format(epoch_resume))
            args.epoch_start = epoch_resume + 1

        # Start training loop
        for epoch in range(args.epoch_start, args.epoch_start + args.n_epochs):
            train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_datasets,
                env,
                tb_logger,
                args
            )
    finally:
        if cleanup_tmp_dir:
            tools.cleanup_tmp_dir(args.tmp_dir)


if __name__ == "__main__":
    run(get_options())
