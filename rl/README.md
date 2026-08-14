# RL — Phase 3a

Reinforcement learning fine-tune. Bot learns to outperform your
imitation baseline by measuring honey/hour and self-improving.

## Files

- `env.py` — Gymnasium environment wrapping Roblox as an RL env
- `reward.py` — multi-timescale reward function (per-tick + per-minute + per-hour + shaped)
- `train_ppo.py` — SB3 PPO training loop

## Running training

Prerequisites:
- Roblox open, logged into the alt (Fredrick)
- Character somewhere useful (a field is fine)
- Pollen bar + honey display visible (HUD reader needs them)
- No user interaction during training (bot drives keys/mouse)

Start:
```
.\.venv\Scripts\python.exe -m rl.train_ppo
```

Press ESC to stop cleanly. Checkpoints save every ~500 seconds.

## Runtime expectations

- 10 FPS ticking = 600 samples/min
- 100k timesteps ≈ 2.5 hours of real-time gameplay
- Full training run: overnight or across multiple sessions
- First few hours: bot behaves randomly — this is expected. Reward starts sparse.

## Monitoring

```
tensorboard --logdir logs/tensorboard
```

Watch: `rollout/ep_rew_mean` should trend up over time.

## Known limits (fixed later)

- Training from scratch — no warm-start from LSTM imitation weights yet.
  Integration with SB3's policy interface is a Phase 3a.5 task.
- Single env (slow) — parallel envs would require running multiple Roblox
  instances which is complex to set up.
- MLP policy — CNN policy would be much better for pixel obs but requires
  a custom SB3 feature extractor.
- Reward function is minimal (honey deltas only) — item pickup bonuses
  from token_values.json get added in Phase 3c.
