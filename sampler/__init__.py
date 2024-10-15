from .uniform_sampler import UniformSampler
from .hard_sampler import HardSampler
from .sampler import Sampler
_Supported_samplers = {
    'uniform': UniformSampler,
    'hard': HardSampler
}

class SamplerFactory:
    @classmethod
    def generate_sampler(cls, 
                        sampler_name, 
                        interactions,
                        model,
                        n_negatives=None,
                        random_seed=1234,
                        **kwargs):
        try:
            spl = _Supported_samplers[sampler_name](sampler_name,
                                                    interactions,
                                                    model, 
                                                    n_negatives,
                                                    random_seed,
                                                    **kwargs)
            return spl
        except KeyError as e:
            raise e('Do not support sampler {}'.format(sampler_name))

__all__ = ['SamplerFactory', 'Sampler']