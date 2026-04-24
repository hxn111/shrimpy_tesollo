from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner


class ShrimpyLowdimRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            **kwargs):
        super().__init__(output_dir)
    
    def run(self, policy: BaseLowdimRunner):
        return dict()
   