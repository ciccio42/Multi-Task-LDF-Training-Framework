from __future__ import annotations
import os
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers import AddedToken
from einops import rearrange, repeat

import vima.nn as vnn
from vima.utils import *
import numpy as np
import cv2
from multi_task_il.models.discrete_logistic import DiscreteMixLogistic

os.environ["TOKENIZERS_PARALLELISM"] = "true"

_kwargs = {
    "single_word": True,
    "lstrip": False,
    "rstrip": False,
    "normalized": True,
}

PLACEHOLDER_TOKENS = [
    AddedToken("{pick_object}", **_kwargs),
]
PLACEHOLDERS = [token.content for token in PLACEHOLDER_TOKENS]
tokenizer = Tokenizer.from_pretrained("t5-base")
tokenizer.add_tokens(PLACEHOLDER_TOKENS)

ENV_OBJECTS = {
    'pick_place': {
        'obj_names': ['greenbox', 'yellowbox', 'bluebox', 'redbox'],
        'ranges': [[0.195, 0.255], [0.045, 0.105], [-0.105, -0.045], [-0.255, -0.195]],
    },
    'nut_assembly': {
        'obj_names': ['nut0', 'nut1', 'nut2'],
        'ranges': [[0.10, 0.31], [-0.10, 0.10], [-0.31, -0.10]]
    }
}


class _DiscreteLogHead(nn.Module):
    def __init__(self, in_dim, out_dim, n_mixtures, const_var=True, sep_var=False):
        super().__init__()
        assert n_mixtures >= 1, "must predict at least one mixture!"
        self._n_mixtures = n_mixtures
        self._dist_size = torch.Size((out_dim, n_mixtures))
        self._mu = nn.Linear(in_dim, out_dim * n_mixtures)
        self._logit_prob = nn.Linear(
            in_dim, out_dim * n_mixtures) if n_mixtures > 1 else None
        if const_var:
            ln_scale = torch.randn(
                out_dim, dtype=torch.float32) / np.sqrt(out_dim)
            self.register_parameter(
                '_ln_scale', nn.Parameter(ln_scale, requires_grad=True))
        if sep_var:
            ln_scale = torch.randn((out_dim, n_mixtures),
                                   dtype=torch.float32) / np.sqrt(out_dim)
            self.register_parameter(
                '_ln_scale', nn.Parameter(ln_scale, requires_grad=True))
        if not (const_var or sep_var):
            self._ln_scale = nn.Linear(in_dim, out_dim * n_mixtures)

    def forward(self, x):  #  x has shape B T d
        mu = self._mu(x).reshape((x.shape[:-1] + self._dist_size))

        if isinstance(self._ln_scale, nn.Linear):
            ln_scale = self._ln_scale(x).reshape(
                (x.shape[:-1] + self._dist_size))
        else:
            ln_scale = self._ln_scale if self.training else self._ln_scale.detach()
            if len(ln_scale.shape) == 1:
                ln_scale = ln_scale.reshape((1, 1, -1, 1)).expand_as(mu)
                # (1, 1, 8, 1) -> (B T, dist_size[0], dist_size[1]) i.e. each mixture has the **same** constant variance
            else:  # the sep_val case:
                ln_scale = repeat(
                    ln_scale, 'out_d n_mix -> B T out_d n_mix', B=x.shape[0], T=x.shape[1])

        logit_prob = self._logit_prob(x).reshape(
            mu.shape) if self._n_mixtures > 1 else torch.ones_like(mu)
        return (mu, ln_scale, logit_prob)


class Policy(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        xf_n_layers: int,
        sattn_n_heads: int,
        xattn_n_heads: int,
        views: list,
        return_dist: bool = True,
        concat_state: bool = False
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.xattn_gpt = vnn.XAttnGPT(
            embed_dim,
            n_layer=xf_n_layers,
            n_head=sattn_n_heads,
            dropout=0.1,
            xattn_n_head=xattn_n_heads,
            xattn_ff_expanding=4,
            xattn_n_positions=256,
            use_geglu=True,
        )

        self.obj_encoder = vnn.ObjEncoder(
            transformer_emb_dim=embed_dim,
            views=views,
            vit_output_dim=768,
            vit_resolution=32,
            vit_patch_size=16,
            vit_width=768,
            vit_layers=4,
            vit_heads=24,
            bbox_mlp_hidden_dim=768,
            bbox_mlp_hidden_depth=2,
        )

        self.end_effector_encoder = vnn.Embedding(
            num_embeddings=2, embedding_dim=2)

        self.obs_fusion_layer = nn.Linear(
            self.obj_encoder.output_dim + 2, embed_dim)

        self.prompt_embedding = vnn.WordEmbedding()
        self.t5_prompt_encoder = vnn.T5PromptEncoder()
        self.t5_prompt_encoder_post_layer = (
            nn.Identity()
            if embed_dim == self.t5_prompt_encoder.output_dim
            else nn.Linear(self.t5_prompt_encoder.output_dim, embed_dim, bias=False)
        )

        self.prompt_obj_post_layer = vnn.build_mlp(
            self.obj_encoder.output_dim,
            hidden_dim=768,
            output_dim=768,
            hidden_depth=2,
        )

        self._views = views
        self._return_dist = return_dist
        self._concat_state = concat_state

        # Action module
        # self._action_module = nn.Sequential(
        #     nn.Linear(, ), nn.ReLU(),
        #     nn.Linear(), nn.ReLU()
        # )
        # self._action_dist = _DiscreteLogHead(
        #     in_dim=,
        #     out_dim=,
        #     n_mixtures=,
        #     const_var=,
        #     sep_var=
        # )

        # Action encoder-decoder
        # self.action_encoder = vnn.ActionEmbedding(
        #     output_dim=embed_dim,
        #     embed_dict={
        #         "pose0_position": vnn.ContinuousActionEmbedding(
        #             output_dim=256,
        #             input_dim=2,
        #             hidden_dim=256,
        #             hidden_depth=1,
        #         ),
        #         "pose0_rotation": vnn.ContinuousActionEmbedding(
        #             output_dim=256,
        #             input_dim=4,
        #             hidden_dim=256,
        #             hidden_depth=1,
        #         ),
        #         "pose1_position": vnn.ContinuousActionEmbedding(
        #             output_dim=256,
        #             input_dim=2,
        #             hidden_dim=256,
        #             hidden_depth=1,
        #         ),
        #         "pose1_rotation": vnn.ContinuousActionEmbedding(
        #             output_dim=256,
        #             input_dim=4,
        #             hidden_dim=256,
        #             hidden_depth=1,
        #         ),
        #     },
        # )
        # self.action_decoder = vnn.ActionDecoder(
        #     input_dim=embed_dim,
        #     action_dims={
        #         "pose0_position": [50, 100],
        #         "pose0_rotation": [50] * 4,
        #         "pose1_position": [50, 100],
        #         "pose1_rotation": [50] * 4,
        #     },
        #     hidden_dim=512,
        #     hidden_depth=2,
        #     activation="relu",
        #     norm_type=None,
        #     last_layer_gain=0.01,
        # )
        # self._n_discrete_x_bins = 50
        # self._n_discrete_y_bins = 100
        # self._n_discrete_z_bins = 50
        # self._n_discrete_rot_bins = 50

    def forward(
        self,
        input: object,
    ):
        B, T, OBJ_NUM, C, W, H = input['obs']['objects']['cropped_img']['front'].shape

        out = dict()
        # 1. Forward prompt assembly
        prompt_tokens, prompt_token_masks = self.forward_prompt_assembly(
            prompts=(input['prompt_token_type'],
                     input['word_batch'],
                     input['image_batch']))
        out['prompt_tokens'] = prompt_tokens
        out['prompt_token_masks'] = prompt_token_masks

        # 2. Forward obs token
        obs_token, obs_mask = self.forward_obs_token(
            input['obs'])

        inference_cache = {}
        predicted_action_batch = None
        for sample in range(B):
            predicted_action_token_trj = None
            obs_token_trajectory = torch.index_select(
                obs_token, 0, torch.tensor(sample).to(obs_token.device))
            obs_mask_trajectory = torch.index_select(
                obs_mask, 0, torch.tensor(sample).to(obs_token.device))
            for t in range(T):

                if t == 0:
                    inference_cache["obs_tokens"] = []
                    inference_cache["obs_masks"] = []
                    inference_cache["action_tokens"] = []

                obs_token_this_step = torch.index_select(
                    obs_token_trajectory, 1, torch.tensor(t).to(obs_token.device))
                obs_mask_this_step = torch.index_select(
                    obs_mask_trajectory, 1, torch.tensor(t).to(obs_token.device))

                # prepare history
                obs_token_this_step = obs_token_this_step.squeeze(0)
                obs_mask_this_step = obs_mask_this_step.squeeze(0)
                inference_cache["obs_tokens"].append(obs_token_this_step[0])
                inference_cache["obs_masks"].append(obs_mask_this_step[0])
                max_objs = max(x.shape[0]
                               for x in inference_cache["obs_tokens"])
                obs_tokens_to_forward, obs_masks_to_forward = [], []
                obs_tokens_this_env, obs_masks_this_env = [], []
                for idx in range(len(inference_cache["obs_tokens"])):
                    obs_this_env_this_step = inference_cache["obs_tokens"][idx]
                    obs_mask_this_env_this_step = inference_cache["obs_masks"][idx]
                    required_pad = max_objs - obs_this_env_this_step.shape[0]
                    obs_tokens_this_env.append(
                        any_concat(
                            [
                                obs_this_env_this_step,
                                torch.zeros(
                                    required_pad,
                                    obs_this_env_this_step.shape[1],
                                    device=obs_token.device,
                                    dtype=obs_this_env_this_step.dtype,
                                ),
                            ],
                            dim=0,
                        )
                    )
                    obs_masks_this_env.append(
                        any_concat(
                            [
                                obs_mask_this_env_this_step,
                                torch.zeros(
                                    required_pad,
                                    device=obs_token.device,
                                    dtype=obs_mask_this_env_this_step.dtype,
                                ),
                            ],
                            dim=0,
                        )
                    )
                obs_tokens_to_forward.append(
                    any_stack(obs_tokens_this_env, dim=0))
                obs_masks_to_forward.append(
                    any_stack(obs_masks_this_env, dim=0))
                obs_tokens_to_forward = any_stack(obs_tokens_to_forward, dim=0)
                obs_masks_to_forward = any_stack(obs_masks_to_forward, dim=0)
                obs_tokens_to_forward = obs_tokens_to_forward.transpose(0, 1)
                obs_masks_to_forward = obs_masks_to_forward.transpose(0, 1)

                if t == 0:
                    action_tokens_to_forward = None
                else:
                    action_tokens_to_forward = any_stack(
                        [any_stack(inference_cache["action_tokens"], dim=0)],
                        dim=0,
                    )
                    action_tokens_to_forward = action_tokens_to_forward.transpose(
                        0, 1)

                obs_token = obs_tokens_to_forward
                obs_mask = obs_masks_to_forward
                action_token = action_tokens_to_forward
                prompt_token = torch.index_select(
                    prompt_tokens, 1, torch.tensor(sample).to(obs_token.device))
                prompt_token_mask = torch.index_select(
                    prompt_token_masks, 0, torch.tensor(sample).to(obs_token.device))
                # 3. Action Token Prediction
                L_obs, B = obs_token.shape[:2]
                L_action = 0 if action_token is None else action_token.shape[0]
                n_max_objs = obs_token.shape[-2]
                L = L_obs * n_max_objs + L_action

                tokens = torch.empty(
                    L, B, self.embed_dim, dtype=torch.float32, device=obs_token.device
                )
                masks = torch.ones(L, B, dtype=torch.bool,
                                   device=obs_token.device)
                obs_token = rearrange(obs_token, "L B Q E -> B L Q E")
                obs_token = rearrange(obs_token, "B L Q E -> B (L Q) E")
                obs_token = rearrange(obs_token, "B L E -> L B E")
                obs_mask = rearrange(obs_mask, "L B Q -> B L Q")
                obs_mask = rearrange(obs_mask, "B L Q -> B (L Q)")
                obs_mask = rearrange(obs_mask, "B L -> L B")
                for q in range(n_max_objs):
                    tokens[q:: n_max_objs + 1] = obs_token[q::n_max_objs]
                    masks[q:: n_max_objs + 1] = obs_mask[q::n_max_objs]
                if action_token is not None:
                    tokens[n_max_objs:: n_max_objs + 1] = action_token

                position_ids = torch.cumsum(masks, dim=0) - 1
                position_ids = position_ids.long()
                prompt_position_ids = torch.cumsum(
                    prompt_token_mask, dim=1) - 1

                tokens_out = self.xattn_gpt(
                    obs_action_tokens=tokens,
                    prompt_tokens=prompt_token,
                    prompt_mask=prompt_token_mask,
                    obs_action_masks=masks.transpose(0, 1),
                    obs_action_position_ids=position_ids.transpose(0, 1),
                    prompt_position_ids=prompt_position_ids,
                )

                predicted_action_token_t = tokens_out[n_max_objs -
                                                      1:: n_max_objs + 1]

                predicted_action_token_t = predicted_action_token_t[-1].unsqueeze(
                    0)
                inference_cache["action_tokens"].append(
                    predicted_action_token_t[0])

                if t == 0:
                    predicted_action_token_trj = predicted_action_token_t
                else:
                    predicted_action_token_trj = torch.cat(
                        (predicted_action_token_trj, predicted_action_token_t), 1)

            if sample == 0:
                predicted_action_batch = predicted_action_token_trj
            else:
                predicted_action_batch = torch.cat(
                    (predicted_action_batch, predicted_action_token_trj), 0)

        out['predicted_action_tokens'] = predicted_action_batch

        if self._return_dist:
            if self._concat_state:
                ac_in = torch.cat((ac_in, input['states']), 2)
            else:
                ac_in = predicted_action_batch

            ac_pred = self._action_module(ac_in)
            mu_bc, scale_bc, logit_bc = self._action_dist(ac_pred)
            out['bc_distrib'] = DiscreteMixLogistic(mu_bc, scale_bc, logit_bc)
        return out

    def forward_prompt_assembly(self, prompts):
        raw_prompts_token_type, word_batch, image_batch = prompts
        batch_word_emb = self.prompt_embedding(word_batch)
        batch_image_emb = self.obj_encoder(**image_batch)
        batch_image_emb = self.prompt_obj_post_layer(batch_image_emb)
        n_max_objs = batch_image_emb.shape[-2]

        L_max = 0
        for raw_prompt in raw_prompts_token_type:
            L_this = 0
            for item in raw_prompt:
                if item == 0:
                    L_this += 1
                elif item == 1:
                    L_this += n_max_objs
                else:
                    raise ValueError(f"Invalid prompt token type {item}")
            L_max = max(L_max, L_this)

        prompt_tokens, prompt_masks = [], []
        for i, raw_prompt in enumerate(raw_prompts_token_type):
            word_ptr, img_ptr = 0, 0
            assembled_prompt = []
            assembled_mask = []
            for item in raw_prompt:
                if item == 0:
                    assembled_prompt.append(batch_word_emb[i][word_ptr])
                    word_ptr += 1
                    assembled_mask.append(True)
                elif item == 1:
                    obj_mask = any_concat(
                        [
                            image_batch["mask"][view][img_ptr]
                            for view in sorted(self._views)
                        ],
                        dim=-1,
                    )
                    for q in range(n_max_objs):
                        assembled_prompt.append(batch_image_emb[i][img_ptr][q])
                        assembled_mask.append(obj_mask[q])
                    img_ptr += 1
                else:
                    raise ValueError(f"Invalid type: {type(item)}")
            num_padding = L_max - len(assembled_prompt)
            assembled_prompt = torch.stack(assembled_prompt, dim=0)
            required_padding = torch.zeros(
                (num_padding, assembled_prompt.shape[1]),
                dtype=torch.float32,
                device=assembled_prompt.device,
            )
            assembled_prompt = torch.cat(
                [assembled_prompt, required_padding], dim=0)
            prompt_tokens.append(assembled_prompt)

            prompt_masks.append(
                torch.cat(
                    [
                        any_to_torch_tensor(
                            assembled_mask,
                            dtype=torch.bool,
                            device=assembled_prompt.device,
                        ),
                        torch.zeros(
                            num_padding,
                            dtype=torch.bool,
                            device=assembled_prompt.device,
                        ),
                    ],
                    dim=0,
                )
            )

        prompt_tokens = torch.stack(prompt_tokens, dim=0)
        prompt_masks = torch.stack(prompt_masks, dim=0)
        prompt_tokens = prompt_tokens.transpose(0, 1)
        if self.t5_prompt_encoder is not None:
            prompt_tokens = self.t5_prompt_encoder(
                prompt_tokens, attention_mask=prompt_masks, batch_first=False
            )
            prompt_tokens = self.t5_prompt_encoder_post_layer(prompt_tokens)
        return prompt_tokens, prompt_masks

    def forward_obs_token(self, obs):
        objects, ee = obs["objects"], obs["ee"]
        B, T, O, _, _, _ = objects['cropped_img']['front'].shape
        leading_dims = ee.shape[:2]

        objects = objects.map_structure(
            func=lambda x: x.reshape(-1, *x.shape[2:]))
        img_feats = self.obj_encoder(**objects)
        img_feats = img_feats.reshape((B, T, O, img_feats.shape[-1]))
        obj_mask = {
            k: objects["mask"][k].reshape(B, T, -1) for k in objects["mask"]
        }

        ee_feats = self.end_effector_encoder(ee)
        ee_feats = ee_feats.unsqueeze(2).repeat(
            1,  img_feats.shape[-3], img_feats.shape[-2], 1)

        obs_feats = self.obs_fusion_layer(
            torch.cat([img_feats, ee_feats], dim=-1))

        obj_mask = any_concat([obj_mask[view]
                               for view in sorted(self._views)], dim=-1)
        return obs_feats, obj_mask

    def forward_action_token(self, action):
        return self.action_encoder(self._de_discretize_actions(action))

    def forward_action_decoder(self, predicted_action_tokens: torch.Tensor):
        return self.action_decoder(predicted_action_tokens)

    def discretize_action(self, action):
        device = action["pose0_position"].device
        boundary_x = torch.linspace(
            start=0, end=1, steps=self._n_discrete_x_bins, device=device
        )
        boundary_y = torch.linspace(
            start=0, end=1, steps=self._n_discrete_y_bins, device=device
        )
        boundary_rot = torch.linspace(
            start=0, end=1, steps=self._n_discrete_rot_bins, device=device
        )

        action["pose0_position"][..., 0] = torch.bucketize(
            action["pose0_position"][..., 0].contiguous(), boundary_x
        )
        action["pose0_position"][..., 1] = torch.bucketize(
            action["pose0_position"][..., 1].contiguous(), boundary_y
        )
        action["pose0_rotation"] = torch.bucketize(
            action["pose0_rotation"].contiguous(), boundary_rot
        )

        action["pose1_position"][..., 0] = torch.bucketize(
            action["pose1_position"][..., 0].contiguous(), boundary_x
        )
        action["pose1_position"][..., 1] = torch.bucketize(
            action["pose1_position"][..., 1].contiguous(), boundary_y
        )
        action["pose1_rotation"] = torch.bucketize(
            action["pose1_rotation"].contiguous(), boundary_rot
        )
        action = {k: v.long() for k, v in action.items()}
        return action

    def _de_discretize_actions(self, actions):
        actions = {k: v.float() for k, v in actions.items()}
        actions["pose0_position"][..., 0] = (
            actions["pose0_position"][..., 0] / self._n_discrete_x_bins
        )
        actions["pose0_position"][..., 1] = (
            actions["pose0_position"][..., 1] / self._n_discrete_y_bins
        )
        actions["pose0_rotation"] = (
            actions["pose0_rotation"] / self._n_discrete_rot_bins
        )

        actions["pose1_position"][..., 0] = (
            actions["pose1_position"][..., 0] / self._n_discrete_x_bins
        )
        actions["pose1_position"][..., 1] = (
            actions["pose1_position"][..., 1] / self._n_discrete_y_bins
        )
        actions["pose1_rotation"] = (
            actions["pose1_rotation"] / self._n_discrete_rot_bins
        )
        return actions


if __name__ == '__main__':
    import debugpy
    debugpy.listen(('0.0.0.0', 5678))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()

    from vima import create_policy_from_ckpt
    ckpt_path = "/home/frosa_loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/models/vima/ckpt/92M.ckpt"
    policy = create_policy_from_ckpt(ckpt_path=ckpt_path, device='cuda:3')
