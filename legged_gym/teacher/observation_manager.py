import torch


class ObsManager:
    def __init__(self, env, obs_groups_cfg):
        self.obs_per_group = {}
        self.obs_dims_per_group = {}
        self.obs_dims_per_group_func = {}
        self.obs = {}
        for group_name, obs_group in obs_groups_cfg.items():
            if obs_group is None:
                continue
            self.obs_per_group[group_name] = []
            obs_dim = 0
            add_noise = obs_group["add_noise"]
            self.obs_dims_per_group_func[group_name] = {}

            for name, params in obs_group.items():
                if not isinstance(params, dict):
                    continue
                if not add_noise:  # turn off all noise
                    params["noise"] = None
                # function = getattr(self, params["func_name"])
                function = params["func"]
                # if function is a string evaluate it, note: it must be imported in the manager module
                if "dofs" in params.keys():
                    params["dof_indices"], _ = env.robot.find_dofs(params["dofs"])
                if "bodies" in params.keys():
                    params["body_indices"], _ = env.robot.find_bodies(params["bodies"])
                self.obs_per_group[group_name].append((function, params))

                delta = function(env, params).shape[1]
                self.obs_dims_per_group_func[group_name][name] = (obs_dim, obs_dim + delta)
                obs_dim += delta

            self.obs_dims_per_group[group_name] = obs_dim

    def compute_obs(self, env):
        self.obs = {}
        for group, function_list in self.obs_per_group.items():
            obs_list = []
            for function, params in function_list:
                obs = function(env, params)
                noise = params.get("noise")
                clip = params.get("clip")
                scale = params.get("scale")
                if noise:
                    obs = self._add_uniform_noise(obs, noise)
                if clip:
                    obs = obs.clip(min=clip[0], max=clip[1])
                if scale is not None:
                    if hasattr(scale, 'shape'):
                      scale = scale[None]
                    obs = scale * obs
                obs_list.append(obs)
            self.obs[group] = torch.cat(obs_list, dim=1)
        return self.obs

    def _add_uniform_noise(self, obs, noise_level):
        return obs + (2 * torch.rand_like(obs) - 1) * noise_level
