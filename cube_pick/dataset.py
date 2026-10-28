"""Student-side loader: NumPy only; no simulator or privileged-state access."""
import json
from pathlib import Path
import numpy as np


class EpisodeDataset:
    """RGB is uint8 (T,V,H,W,C); normalize/transpose in the student model."""
    def __init__(self, root, split="train"):
        self.root = Path(root)
        self.manifest = json.loads((self.root/"manifest.json").read_text())
        self.records = [r for r in self.manifest["episodes"] if r["split"] == split]
        if not self.records:
            raise ValueError(f"Empty or unknown split: {split}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with np.load(self.root/self.records[index]["observations"], allow_pickle=False) as data:
            return {k: data[k] for k in ("rgb", "proprio", "actions", "timestamps", "goal_xy")}

    def window(self, index, start, context=4, horizon=5):
        """Last context observation + future actions -> future observations.

        Samples never cross an episode boundary; future targets are separate
        from inputs. All paired alternative rollouts stay in the same split.
        """
        data = self[index]
        anchor = start + context - 1
        if start < 0 or context < 1 or horizon < 1 or anchor+horizon >= len(data["rgb"]):
            raise IndexError("Window exceeds this episode")
        return {
            "inputs": {"rgb": data["rgb"][start:anchor+1],
                       "proprio": data["proprio"][start:anchor+1],
                       "past_actions": data["actions"][start:anchor],
                       "future_actions": data["actions"][anchor:anchor+horizon],
                       "goal_xy": data["goal_xy"]},
            "targets": {"rgb": data["rgb"][anchor+1:anchor+horizon+1],
                        "proprio": data["proprio"][anchor+1:anchor+horizon+1]},
        }
