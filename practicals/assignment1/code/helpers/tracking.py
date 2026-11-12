"""Optional Weights & Biases logging. Local metrics never depend on this tool."""
import json
from pathlib import Path


class RunTracker:
    def __init__(self, args):
        self.project = getattr(args, "wandb_project", None)
        self.name = getattr(args, "wandb_name", None)
        if self.name and not self.project:
            raise ValueError("--wandb-name requires --wandb-project")
        self.run = None
        self.wandb = None
        if self.project:
            try:
                import wandb
            except ImportError as error:
                raise ImportError("Optional W&B logging needs: uv pip install wandb") from error
            self.wandb = wandb

    def __enter__(self):
        return self

    def start(self, directory, settings):
        if self.wandb is None:
            return
        path = Path(directory) / "wandb-run.json"
        self.run = self.wandb.init(
            project=self.project,
            name=self.name or Path(directory).name,
            config=settings,
            dir=str(directory),
        )
        self.run.define_metric("tokens")
        self.run.define_metric("*", step_metric="tokens")
        path.write_text(json.dumps({"project": self.project, "id": self.run.id}) + "\n")

    def log(self, row):
        if self.run is not None:
            self.run.log({key: value for key, value in row.items() if value is not None})

    def __exit__(self, kind, value, traceback):
        if self.run is not None:
            self.run.finish(exit_code=0 if kind is None else 1)
