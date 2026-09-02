"""ATRL - Anchored Transitive RL.
"""
import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections as mlc
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import (
    GCActor,
    GCDiscreteActor,
    GCDiscreteCritic,
    GCValue,
    ActorVectorField,
)


class ATRLAgent(flax.struct.PyTreeNode):

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def bce_loss(pred_logit, target):
        log_pred = jax.nn.log_sigmoid(pred_logit)
        log_not_pred = jax.nn.log_sigmoid(-pred_logit)
        loss = -(log_pred * target + log_not_pred * (1 - target))
        return loss

    def critic_loss(self, batch, grad_params):
        discount = self.config["discount"]

        goal_key = (
            "value_goal_observations"
            if self.config["oracle_distill"]
            else "value_goals"
        )
        midpoint_goal_key = (
            "value_midpoint_observations"
            if self.config["oracle_distill"]
            else "value_midpoint_goals"
        )

        q_logits = self.network.select("critic")(
            batch["observations"],
            goals=batch[goal_key],
            actions=batch["actions"],
            params=grad_params,
        )  # [num_ensembles, batch]
        qs = jax.nn.sigmoid(q_logits)

        # ---- divide-and-conquer target --------------------
        first_q_logits = self.network.select("target_critic")(
            batch["observations"],
            goals=batch[midpoint_goal_key],
            actions=batch["actions"],
        )
        first_q = jnp.where(
            (batch["value_midpoint_offsets"] <= 1)[None, ...],
            discount ** batch["value_midpoint_offsets"][None, ...],
            jax.nn.sigmoid(first_q_logits),
        )

        second_q_logits = self.network.select("target_critic")(
            batch["value_midpoint_observations"],
            goals=batch[goal_key],
            actions=batch["value_midpoint_actions"],
        )
        second_offset = batch["value_offsets"] - batch["value_midpoint_offsets"]
        second_q = jnp.where(
            (second_offset <= 1)[None, ...],
            discount ** second_offset[None, ...],
            jax.nn.sigmoid(second_q_logits),
        )
        dc_target = first_q * second_q

        # ---- one-step TD anchor ----------------------------------------
        next_q = jax.nn.sigmoid(
            self.network.select("target_critic")(
                batch["next_observations"],
                goals=batch[goal_key],
                actions=batch["next_actions"],
            )
        )
        td_target = jnp.where(
            batch["goal_is_cur"][None, ...],
            1.0,  # already at the goal: hitting time 0
            jnp.where(
                batch["goal_is_next"][None, ...],
                discount,  # one step away: TRL's `offset <= 1` base case
                discount * next_q,
            ),
        )

        # ---- select, and weight -----------------------------------------
        decomposable = batch["value_decomposable"][None, ...]

        # ---- occupancy-ratio hindsight weight ---------------------------
        cur_q_target = jax.nn.sigmoid(
            self.network.select("target_critic")(
                batch["observations"],
                goals=batch[goal_key],
                actions=batch["actions"],
            )
        )
        hw_eps = 1e-6
        raw_hw = cur_q_target / jnp.maximum(discount * next_q, hw_eps)

        clip_c = self.config.get("hindsight_weight_clip", 1.0)
        clipped_hw = jnp.clip(raw_hw, 1.0 / (1.0 + clip_c), 1.0 + clip_c)

        dec_w = decomposable.astype(clipped_hw.dtype)  # [1, batch]
        n_dec_rows = dec_w.sum(axis=-1, keepdims=True)  # [1, 1]
        hw_sum = (clipped_hw * dec_w).sum(axis=-1, keepdims=True)  # [ens, 1]
        hw_mean_dec = jnp.where(
            n_dec_rows > 0.5,
            hw_sum / jnp.maximum(n_dec_rows, 1.0),
            1.0,
        )  # [ens, 1]
        norm_hw = clipped_hw / jnp.maximum(hw_mean_dec, hw_eps)

        hindsight_weight = jax.lax.stop_gradient(
            jnp.where(decomposable, norm_hw, 1.0)
        )
        use_hw = self.config.get("use_hindsight_weight", False)
        applied_hw = (
            hindsight_weight if use_hw else jnp.ones_like(hindsight_weight)
        )

        anchored_target = jnp.maximum(dc_target, td_target)
        anchor_lift = anchored_target - dc_target  # >= 0 by construction

        if self.config.get("use_anchored_target", False):
            row_target = anchored_target
        else:
            row_target = dc_target

        target = jnp.where(decomposable, row_target, td_target)

        expectile = jnp.where(
            decomposable, self.config["expectile"], self.config["td_expectile"]
        )
        expectile_weight = jnp.where(target >= qs, expectile, 1 - expectile)

        dist = jax.lax.stop_gradient(jnp.log(target) / jnp.log(discount))
        dist_weight = (1 / (1 + dist)) ** self.config["lam"]

        row_weight = batch["value_target_valid"][None, ...]

        q_loss = (
            expectile_weight * dist_weight * row_weight * applied_hw * self.bce_loss(q_logits, target)
        )
        total_loss = q_loss.mean()

        if self.config["oracle_distill"]:
            distill_q_logits = self.network.select("oracle_critic")(
                batch["observations"],
                goals=batch["value_goals"],
                actions=batch["actions"],
                params=grad_params,
            )
            distill_loss = self.bce_loss(
                distill_q_logits, jax.lax.stop_gradient(qs)
            ).mean()

            total_loss = total_loss + distill_loss

        dec_f = batch["value_decomposable"].astype(jnp.float32)  # [batch]
        n_dec = jnp.maximum(dec_f.sum(), 1.0)
        n_td = jnp.maximum((1.0 - dec_f).sum(), 1.0)
        n_ens = q_logits.shape[0]

        dec_b = dec_f[None, ...]  # [1, batch]
        hw_denom = n_dec * n_ens
        hw_mean = (hindsight_weight * dec_b).sum() / hw_denom
        hw_log = jnp.log(jnp.maximum(hindsight_weight, hw_eps))
        hw_log_mean = (hw_log * dec_b).sum() / hw_denom
        hw_log_var = (((hw_log - hw_log_mean) ** 2) * dec_b).sum() / hw_denom
        hw_clip_frac = (
            (
                (raw_hw < 1.0 / (1.0 + clip_c)) | (raw_hw > 1.0 + clip_c)
            ).astype(jnp.float32)
            * dec_b
        ).sum() / hw_denom
        hw_raw_gmean = jnp.exp(
            (jnp.log(jnp.maximum(raw_hw, hw_eps)) * dec_b).sum() / hw_denom
        )
        hw_min = jnp.min(jnp.where(decomposable, hindsight_weight, jnp.inf))
        hw_max = jnp.max(jnp.where(decomposable, hindsight_weight, -jnp.inf))

        return total_loss, {
            "total_loss": total_loss,
            "q_loss": q_loss.mean(),
            "q_mean": qs.mean(),
            "q_max": qs.max(),
            "q_min": qs.min(),
            "target_mean": target.mean(),
            "dc_target_mean": (dc_target * dec_f[None, ...]).sum() / (n_dec * n_ens),
            "td_target_mean": (td_target * (1.0 - dec_f)[None, ...]).sum()
            / (n_td * n_ens),
            "frac_decomposable": dec_f.mean(),
            "frac_row_valid": batch["value_target_valid"].mean(),
            "anchor_frac_lifted": (
                (anchor_lift > 0).astype(jnp.float32) * dec_f[None, ...]
            ).sum()
            / (n_dec * n_ens),
            "anchor_lift_mean": (anchor_lift * dec_f[None, ...]).sum()
            / (n_dec * n_ens),
            "use_anchored_target": jnp.asarray(
                1.0 if self.config.get("use_anchored_target", False) else 0.0,
                dtype=jnp.float32,
            ),
            "hw_mean": hw_mean,
            "hw_gmean": jnp.exp(hw_log_mean),
            "hw_sd_log": jnp.sqrt(jnp.maximum(hw_log_var, 0.0)),
            "hw_min": hw_min,
            "hw_max": hw_max,
            "hw_frac_clipped": hw_clip_frac,
            "hw_raw_gmean": hw_raw_gmean,
            "use_hindsight_weight": jnp.asarray(
                1.0 if use_hw else 0.0, dtype=jnp.float32
            ),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss. Identical to TRL."""
        pe_info = self.config[self.config["pe_type"]]

        if self.config["pe_type"] == "rpg":
            dist = self.network.select("actor")(
                batch["observations"], batch["actor_goals"], params=grad_params
            )
            if pe_info["const_std"]:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            critic_module = (
                "oracle_critic" if self.config["oracle_distill"] else "critic"
            )
            q1, q2 = self.network.select(critic_module)(
                batch["observations"], batch["actor_goals"], q_actions
            )
            q = jnp.minimum(q1, q2)

            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            log_prob = dist.log_prob(batch["actions"])

            bc_loss = -(pe_info["alpha"] * log_prob).mean()

            actor_loss = q_loss + bc_loss

            return actor_loss, {
                "actor_loss": actor_loss,
                "q_loss": q_loss,
                "bc_loss": bc_loss,
                "q_mean": q.mean(),
                "q_abs_mean": jnp.abs(q).mean(),
                "bc_log_prob": log_prob.mean(),
                "mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
                "std": jnp.mean(dist.scale_diag),
            }

        elif self.config["pe_type"] == "discrete":
            dist = self.network.select("actor")(
                batch["observations"], batch["actor_goals"], params=grad_params
            )

            n_actions = jnp.repeat(
                jnp.expand_dims(jnp.arange(0, pe_info["action_ct"]), 1),
                self.config["batch_size"],
                axis=1,
            )
            n_obs = jnp.repeat(
                jnp.expand_dims(batch["observations"], 0), pe_info["action_ct"], axis=0
            )
            n_goals = jnp.repeat(
                jnp.expand_dims(batch["actor_goals"], 0), pe_info["action_ct"], axis=0
            )

            q = self.network.select(
                "oracle_critic" if self.config["oracle_distill"] else "critic"
            )(n_obs, n_goals, n_actions).mean(axis=0)

            v = jnp.sum(q * dist.probs.T, axis=0)
            q_loss = -v.mean()

            log_prob = dist.log_prob(batch["actions"])
            bc_loss = -(pe_info["alpha"] * log_prob).mean()

            actor_loss = q_loss + bc_loss

            return actor_loss, {
                "actor_loss": actor_loss,
                "q_loss": q_loss,
                "bc_loss": bc_loss,
                "q_mean": q.mean(),
                "q_abs_mean": jnp.abs(q).mean(),
                "bc_log_prob": log_prob.mean(),
            }

        elif self.config["pe_type"] == "frs":
            batch_size, action_dim = batch["actions"].shape
            x_rng, t_rng = jax.random.split(rng, 2)

            x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
            x_1 = batch["actions"]
            t = jax.random.uniform(t_rng, (batch_size, 1))
            x_t = (1 - t) * x_0 + t * x_1
            y = x_1 - x_0

            pred = self.network.select("actor")(
                batch["observations"], batch["actor_goals"], x_t, t, params=grad_params
            )

            actor_loss = jnp.mean((pred - y) ** 2)

            return actor_loss, {"actor_loss": actor_loss}

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "critic")

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        pe_info = self.config[self.config["pe_type"]]

        if self.config["pe_type"] == "frs":
            n_observations = jnp.repeat(
                jnp.expand_dims(observations, 0), pe_info["num_samples"], axis=0
            )
            n_goals = jnp.repeat(
                jnp.expand_dims(goals, 0), pe_info["num_samples"], axis=0
            )

            n_actions = jax.random.normal(
                seed,
                (
                    pe_info["num_samples"],
                    *observations.shape[:-1],
                    self.config["action_dim"],
                ),
            )
            for i in range(pe_info["flow_steps"]):
                t = jnp.full(
                    (pe_info["num_samples"], *observations.shape[:-1], 1),
                    i / pe_info["flow_steps"],
                )
                vels = self.network.select("actor")(
                    n_observations, n_goals, n_actions, t
                )
                n_actions = n_actions + vels / pe_info["flow_steps"]
            n_actions = jnp.clip(n_actions, -1, 1)

            critic_module = (
                "oracle_critic" if self.config["oracle_distill"] else "critic"
            )
            q = self.network.select(critic_module)(
                n_observations, goals=n_goals, actions=n_actions
            )

            if len(observations.shape) == 2:
                actions = n_actions[
                    jnp.argmax(q, axis=0), jnp.arange(observations.shape[0])
                ]
            else:
                actions = n_actions[jnp.argmax(q)]

            return actions

        else:
            dist = self.network.select("actor")(
                observations, goals, temperature=temperature
            )
            actions = dist.sample(seed=seed)

            if self.config["pe_type"] != "discrete":
                actions = jnp.clip(actions, -1, 1)

            return actions

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        if config.get("use_hindsight_weight", False) and not config["dataset"][
            "value_geom_sample"
        ]:
            raise ValueError(
                "use_hindsight_weight=True requires "
                "dataset.value_geom_sample=True. The occupancy-ratio weight "
                "assumes goals are drawn with offsets ~ Geometric(1-discount); "
                "with uniform sampling the hindsight goal density is not the "
                "discounted occupancy and the critic is not a valid stand-in "
                "for it. Either set value_geom_sample=True or set "
                "use_hindsight_weight=False."
            )
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch["observations"]
        ex_actions = example_batch["actions"]
        ex_goals = example_batch["actor_goals"]
        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]
        pe_info = config[config["pe_type"]]

        if config["pe_type"] == "discrete":
            critic_def = GCDiscreteCritic(
                hidden_dims=config["value_hidden_dims"],
                layer_norm=config["layer_norm"],
                num_ensembles=2,
                action_dim=config["discrete"]["action_ct"],
            )
            oracle_critic_def = GCDiscreteCritic(
                hidden_dims=config["value_hidden_dims"],
                layer_norm=config["layer_norm"],
                num_ensembles=2,
                action_dim=config["discrete"]["action_ct"],
            )
        else:
            critic_def = GCValue(
                hidden_dims=config["value_hidden_dims"],
                layer_norm=config["layer_norm"],
                num_ensembles=2,
            )
            oracle_critic_def = GCValue(
                hidden_dims=config["value_hidden_dims"],
                layer_norm=config["layer_norm"],
                num_ensembles=2,
            )

        if config["pe_type"] == "frs":
            actor_def = ActorVectorField(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=action_dim,
                layer_norm=config["layer_norm"],
            )
            ex_actor_in = (ex_observations, ex_goals, ex_actions, ex_times)
        elif config["pe_type"] == "discrete":
            actor_def = GCDiscreteActor(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=config["discrete"]["action_ct"],
                layer_norm=config["layer_norm"],
            )
            ex_actor_in = (ex_observations, ex_goals, ex_actions)
        else:
            actor_def = GCActor(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=action_dim,
                layer_norm=config["layer_norm"],
                state_dependent_std=False,
                const_std=pe_info["const_std"],
            )
            ex_actor_in = (ex_observations, ex_goals, ex_actions)

        ex_critic_goals = ex_observations if config["oracle_distill"] else ex_goals
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_critic_goals, ex_actions)),
            target_critic=(
                copy.deepcopy(critic_def),
                (ex_observations, ex_critic_goals, ex_actions),
            ),
            oracle_critic=(oracle_critic_def, (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, ex_actor_in),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]

        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params["modules_target_critic"] = params["modules_critic"]

        config["action_dim"] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = mlc.ConfigDict(
        dict(
            agent_name="atrl",
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(1024,) * 4,
            value_hidden_dims=(1024,) * 4,
            layer_norm=True,
            discount=0.999,
            tau=0.005,
            lam=0.0,
            expectile=0.7,  
            oracle_distill=False,
            pe_type="frs",
            frs=mlc.ConfigDict(dict(flow_steps=10, num_samples=32)),
            rpg=mlc.ConfigDict(dict(alpha=0.03, const_std=True)),
            discrete=mlc.ConfigDict(dict(alpha=0.03, action_ct=0)),
            # ---- ATRL ---------------------------------------------------
            td_expectile=0.5,
            use_anchored_target=False,
            use_hindsight_weight=True,
            hindsight_weight_clip=1.0,
            dataset=mlc.ConfigDict(
                dict(
                    dataset_class="GCDataset",
                    value_p_curgoal=0.0,
                    value_p_trajgoal=0.7,
                    value_p_randomgoal=0.3,
                    value_geom_sample=True,
                    actor_p_curgoal=0.0,
                    actor_p_trajgoal=0.5,
                    actor_p_randomgoal=0.5,
                    actor_geom_sample=True,
                )
            ),
        )
    )
    return config
