from .encoder import EMGEncoder
from .predictor import Predictor
from .joiner import Joiner
from .transducer import Transducer


def build_transducer(config: dict) -> Transducer:
    """Build transducer model from config dictionary."""
    encoder_config = config['model']['encoder']
    predictor_config = config['model']['predictor']
    joiner_config = config['model']['joiner']

    encoder = EMGEncoder(
        emg_channels=config['data'].get('num_channels', 8),
        num_sessions=encoder_config.get('num_sessions', 8),
        session_embed_dim=encoder_config.get('session_embed_dim', 32),
        conv_dim=encoder_config.get('conv_channels', 768),
        num_conv_blocks=encoder_config.get('num_conv_blocks', 3),
        d_model=encoder_config.get('d_model', 768),
        nhead=encoder_config.get('num_heads', 8),
        dim_feedforward=encoder_config.get('ff_dim', 2048),
        num_layers=encoder_config.get('num_layers', 6),
        dropout=encoder_config.get('dropout', 0.1),
        output_dim=encoder_config.get('output_dim', 128),
    )

    predictor = Predictor(
        vocab_size=predictor_config.get('vocab_size', 43),
        embed_dim=predictor_config.get('embed_dim', 128),
        hidden_dim=predictor_config.get('hidden_dim', 320),
        num_layers=predictor_config.get('num_layers', 1),
        output_dim=predictor_config.get('output_dim', 128),
    )

    joiner = Joiner(
        input_dim=joiner_config.get('input_dim', 128),
        vocab_size=joiner_config.get('vocab_size', 43),
    )

    return Transducer(encoder, predictor, joiner, blank_id=0)


__all__ = ["EMGEncoder", "Predictor", "Joiner", "Transducer", "build_transducer"]
