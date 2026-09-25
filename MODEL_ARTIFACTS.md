# Model artifacts (RL Baseline v0.1)

The trained PPO model and its VecNormalize statistics are **not stored in git** and there is **no shared download
location yet**. They currently exist only outside the repository, in the maintainer's local training-output directory
(`mujoco_rl/runs/` of the original `drone_stack` checkout; the files there are root-owned). This file
defines which files form the baseline, how the code finds them, and how to verify them. Ask the maintainer for the two
files below, or restore them from wherever the team decides to host them.

## The pair of record

A PPO model is only valid together with the VecNormalize statistics it was trained with (observation normalization, 16-D).
They are one artifact. Never mix generations.

| role | file name | size | SHA256 |
|---|---|---|---|
| PPO model | `ppo_landing_residual_v4_stage2_contact_final.zip` | 488,828 B | `9649fcd5e669c099193f9fd05687c22190625642fa373aedb530a49ef8b9949b` |
| VecNormalize | `vecnormalize_v4_stage2_contact.pkl` | 2,109 B | `de32e096d7d778a50f06d996f1e8bcd7c340f97fd015348fce108237d23bbfe3` |

Both were last modified 2026-06-30 14:59 (file times of the maintainer's copies). Hashes computed 2026-09-25.

Facts read from the files themselves: obs space (16,), action space (3,), `num_timesteps` 2,523,136, VecNormalize
`obs_rms.count` 2,523,148, `clip_obs` 10, `gamma` 0.995. PPO hyperparameters stored in the checkpoint: learning rate 1e-4,
`n_steps` 2048, batch 256, gamma 0.995, `ent_coef` 0, policy/value networks 2 x 128, `log_std_init` -2. The training
script that created the stage-0 model (and therefore the `PPO(...)` constructor call) is not in the repository.

## Where the code looks

Resolution order, implemented once in `landing_rl/evaluation/artifacts.py`:

1. `--model` / `--vecnorm` on the evaluation script (always wins)
2. environment variable `UGRP_RL_ARTIFACT_DIR` (a directory containing the two files above)
3. `mujoco_rl/runs/` inside this checkout (git-ignored)

None of this depends on the current working directory. Results are written by default to `mujoco_rl/runs/paper_eval`
(also git-ignored), never into `UGRP_RL_ARTIFACT_DIR`, so an archive directory cannot be overwritten by accident.

```bash
# option A: put both files into mujoco_rl/runs/
mkdir -p mujoco_rl/runs && cp /path/to/{ppo_landing_residual_v4_stage2_contact_final.zip,vecnormalize_v4_stage2_contact.pkl} mujoco_rl/runs/

# option B: leave them anywhere and point to them
export UGRP_RL_ARTIFACT_DIR=/path/to/dir
# option C: per command
python -m mujoco_rl.eval_compare_v2 --policy ppo --model /path/M.zip --vecnorm /path/V.pkl
```

Verify after copying:

```bash
cd "${UGRP_RL_ARTIFACT_DIR:-mujoco_rl/runs}" && sha256sum -c <<'EOF'
9649fcd5e669c099193f9fd05687c22190625642fa373aedb530a49ef8b9949b  ppo_landing_residual_v4_stage2_contact_final.zip
de32e096d7d778a50f06d996f1e8bcd7c340f97fd015348fce108237d23bbfe3  vecnormalize_v4_stage2_contact.pkl
EOF
```

## Git-ignore status

`.gitignore` excludes `mujoco_rl/runs/`, `mujoco_rl/*.zip`, `mujoco_rl/**/*.zip` and, globally, `*.pt *.pth *.onnx *.engine
*.tflite`. Do not commit the model. If the team later adopts Git LFS or release assets, update this file and
`landing_rl/evaluation/artifacts.py` together.

## Tests that need the artifacts

`landing_rl/tests/test_checkpoint_compatibility.py` loads the pair of record against the legacy and the modular
environment and compares deterministic rollouts. Without the artifacts it **skips with a message naming the missing files**.
Set `UGRP_RL_REQUIRE_ARTIFACTS=1` to make missing artifacts a failure instead of a skip (use it for release checks):

```bash
UGRP_RL_ARTIFACT_DIR=/path/to/dir UGRP_RL_REQUIRE_ARTIFACTS=1 \
    python -m unittest discover -s landing_rl/tests -p "test_*.py"
```

## Other artifacts that exist (historical, not the baseline)

Each model pairs only with the VecNormalize file of the same name. `num_timesteps` and the VecNormalize sample counts
show the lineage of these files (see `BASELINE.md`, "Training lineage").

| model | VecNormalize | model `num_timesteps` | notes |
|---|---|---|---|
| `ppo_landing_residual_v3_stage0_final.zip` | `vecnormalize_v3_stage0.pkl` | 507,904 | stage 0; no training script in the repo |
| `ppo_landing_residual_v3_stage1_final.zip` | `vecnormalize_v3_stage1.pkl` | 1,515,520 | `train_ppo_v3_long.py` (from stage 0) |
| `ppo_landing_residual_v3_stage1_rewardfix1_final.zip` | `vecnormalize_v3_stage1_rewardfix1.pkl` | 1,515,520 | `train_ppo_stage1_1m.py` (from stage 0) |
| `ppo_landing_residual_v3_stage2_contact_final.zip` | `vecnormalize_v3_stage2_contact.pkl` | 1,515,520 | contact-enabled; no script in the repo |
| `ppo_landing_residual_v4_stage2_contact_final.zip` | `vecnormalize_v4_stage2_contact.pkl` | 2,523,136 | **baseline**; `train_ppo_stage2_contact.py` (from the stage-2 v3 pair) |

SHA256 of the historical files (for `sha256sum -c`):

```
fb631dff48abf04f6b5baf7ce74af4ba1a5d6f05b46a36e7eca526e9b03cf6ee  ppo_landing_residual_v3_stage0_final.zip
bb547f8f0cbfdaaaa357eed8731e9348f702006219e9203bc312431362a7991d  vecnormalize_v3_stage0.pkl
a47656a8a1f4427ce687c939dd1ed18098c90af3696702b8d7dccf3911dde4b5  ppo_landing_residual_v3_stage1_final.zip
d39634dea19badaaa4ae8f0feedd42062b9bb93244edbf7af3f7e296d5e47176  vecnormalize_v3_stage1.pkl
9825cc11c559da61f8578946ebc47cf3030901aa3e4ac5a97372ceec5f76d38c  ppo_landing_residual_v3_stage1_rewardfix1_final.zip
a76d17d7baec2eeb887078a14ce539d5933b16eda89ed86d07ae593b58d4f871  vecnormalize_v3_stage1_rewardfix1.pkl
b1d6165a3b018fdbefe2d00a19d7a5f60edfafd9699a899b6e6250513d264deb  ppo_landing_residual_v3_stage2_contact_final.zip
9ea4e13ecf208058062f4107ca0e0070a42afba8672f42840024276b8af0c2df  vecnormalize_v3_stage2_contact.pkl
```

Why pairing matters: the normalization statistics differ substantially between generations (for example the position-error
standard deviations are about 0.5-0.8 m in stage 0 and about 1.4-1.7 m in the v4 pair, and an invalid target
(`target_valid` = 0) normalizes to -10.0 with the stage-0/1 statistics versus -4.9 with the v4 statistics). Nothing inside the `.zip` or `.pkl` records which file
belongs to which, so the file names are the only pairing information.

## Archived reference evaluation outputs (not in git)

Produced on 2026-08-31 with the pair of record, before the entry-point migration. Reproduced by the current scripts on
2026-09-25 (see `BASELINE.md`).

| file | SHA256 |
|---|---|
| `eval_compare_v2_paper_200ep.txt` | `0e1a538430bfc2c65e5580dbf97990de1e35475f327839684d241d1549ed53d0` |
| `paper_eval/robustness_summary.csv` | `3908f12bb21e0c840089b7117f7f2554348708b4dbd51c0619d59c16448ffeda` |
| `paper_eval/robustness_config.txt` | `ef8f6fde9959c6b71745a26deb178089eba6f14a58d6702d5a15922628abd3d6` |
