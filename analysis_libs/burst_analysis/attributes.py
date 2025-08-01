from dataclasses import dataclass
from typing import Any

@dataclass
class NeuronAttributes:
    cluster_id: int
    channel: int
    position: Any
    template: Any
    amplitudes: Any
    waveforms: Any
    neighbor_channels: Any
    neighbor_positions: Any
    neighbor_templates: Any

    