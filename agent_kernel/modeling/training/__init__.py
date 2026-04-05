from .hybrid_dataset import build_hybrid_training_examples, materialize_hybrid_training_dataset
from .corpus_acquisition import external_corpus_examples, materialize_external_corpus
from .qwen_dataset import materialize_qwen_sft_dataset
from .universal_dataset import collect_universal_decoder_examples, materialize_universal_decoder_dataset

__all__ = [
    "build_hybrid_training_examples",
    "collect_universal_decoder_examples",
    "external_corpus_examples",
    "materialize_hybrid_training_dataset",
    "materialize_external_corpus",
    "materialize_qwen_sft_dataset",
    "materialize_universal_decoder_dataset",
    "train_hybrid_decoder_from_universal_dataset",
    "train_hybrid_runtime_from_dataset",
    "train_hybrid_runtime_from_trajectories",
]


def train_hybrid_runtime_from_dataset(*args, **kwargs):
    from .hybrid_trainer import train_hybrid_runtime_from_dataset as _impl

    return _impl(*args, **kwargs)


def train_hybrid_runtime_from_trajectories(*args, **kwargs):
    from .hybrid_trainer import train_hybrid_runtime_from_trajectories as _impl

    return _impl(*args, **kwargs)


def train_hybrid_decoder_from_universal_dataset(*args, **kwargs):
    from .universal_trainer import train_hybrid_decoder_from_universal_dataset as _impl

    return _impl(*args, **kwargs)
