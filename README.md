# Abductive World Modeling (AWM)

**Abductive World Modeling via Causal Representation Learning** uses a predicted future to help explain the present. Rather than treating a video representation as an undifferentiated feature, AWM considers both the current observation and its predicted future, then infers a structured state that can account for how the scene evolves.

AWM organises the inferred state as a **Hierarchical Abductive State Pyramid (HASP)**, distinguishing three complementary aspects of a scene: **entities** (what exists), **dynamics** (how those entities change), and **relations** (how they interact). This separation makes future evidence useful for reasoning about objects and their behavior, not only for predicting a single future representation.

AWM is intended to support physical outcome prediction, event reasoning, and action understanding. Its structured states also make it possible to examine which entities, changes, and interactions contribute to a prediction. These are the capabilities the method aims to provide, not a claim that every prediction is causally identified or correct.

## Reproduction

For prerequisites, data and model preparation, and reproduction instructions, see [docs/exp/README.md](docs/exp/README.md).

## License

This repository is released under the Apache License 2.0; see [LICENSE](LICENSE). Vendored V-JEPA 2 code retains its upstream license in [external/vjepa2/LICENSE](external/vjepa2/LICENSE).
