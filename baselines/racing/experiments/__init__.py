

def dispatch_experiment(args, logdir):
    if args.agent in ['mpo', 'd4pg', 'lstm-mpo']:
        from baselines.racing.experiments.acme import make_experiment
    elif args.agent in ['sac', 'ppo', 'lstm-ppo']:
        from baselines.racing.experiments.sb3 import make_experiment
    else:
        raise NotImplementedError(args.agent)

    return make_experiment(args, logdir)