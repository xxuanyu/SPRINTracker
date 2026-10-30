# Architecture and code correspondence

The supplied figure is the author-provided conceptual model diagram. The executable implementation is authoritative for reproduction.

- Deformable DETR supplies detections and appearance representations.
- BPMA embeds a 16-dimensional fish state and uses a causal motion Transformer. The Gaussian head uses diagonal covariance. Predicted uncertainty contributes an additive Mahalanobis-distance association bias.
- ORPSM uses persistent identity slots, short-term dynamics, and gated identity-memory updates. In this implementation, memory is reconstructed within the configured history window; the diagram should not be read as unlimited storage across an entire video. Slot-state terminology in code includes visible, occluded, and not-yet-observed states.
- STRID constructs local neighborhood graphs and applies shared interaction encoding to current and historical tokens. Relational information augments identity decoding through features and attention biases.
- Current detection fusion includes appearance, shape, and tank geometry. Historical fusion additionally includes motion features.
- The global decoder uses current queries and historical keys/values. Temporal, motion, and relational biases enter the attention computation. The first decoder layer omits self-attention in the underlying MOTIP design.
- The diagram's relation loss corresponds to the interaction objective in code. Loss weights and active components are controlled by YAML configuration.

Main files are `models/motip/motip.py`, `trajectory_modeling.py`, `id_decoder.py`, `fish_motion.py`, `persistent_memory.py`, `fish_interaction.py`, and `models/runtime_tracker.py`.
