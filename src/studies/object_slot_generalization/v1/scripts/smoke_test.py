"""CPU/GPU smoke test for the isolated object-slot module."""
import argparse
import torch
from src.studies.object_slot_generalization.v1.model import ObjectSlotModel, slot_losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--input-dim", type=int, default=1280)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--patches", type=int, default=256)
    a = p.parse_args()
    d = torch.device(a.device)
    model = ObjectSlotModel(a.input_dim, a.hidden_dim).to(d)
    x = torch.randn(a.batch, a.frames, a.patches, a.input_dim, device=d)
    out = model(x)
    loss = slot_losses(out, target_features=x)["total"]
    loss.backward()
    print({"device": str(d), "input": list(x.shape), "slots": list(out.slots.shape),
           "assignment": list(out.assignment.shape), "loss": float(loss.detach())})


if __name__ == "__main__":
    main()
