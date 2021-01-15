MPO = dict(
    policy_layers=[400, 400],
    critic_layers=[400, 400],
    policy_lr=1e-4,
    critic_lr=1e-4,
    dual_lr=1e-4,
    loss_params=dict(
        epsilon=1e-1,
        epsilon_penalty=1e-3,
        epsilon_mean=1e-3,
        epsilon_stddev=1e-6,
        init_log_temperature=1.,
        init_log_alpha_mean=1.,
        init_log_alpha_stddev=10.
    ),
    discount=0.99,
    batch_size=256,
    target_policy_update_period=100,
    target_critic_update_period=100,
    samples_per_insert=32.0,
    n_step=5,
    num_samples=20,
    clipping=True,
)
