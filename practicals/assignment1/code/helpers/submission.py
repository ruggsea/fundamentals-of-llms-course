"""Package two course runs, publish explicitly, and check a pinned Hub submission.

Package/check execute the selected saved Python source. Use only your own runs.
The Hub is file hosting here; no Transformers conversion or inference API is used.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile

import torch

FORMAT = "course-submission-v1"
ROLES = ("baseline", "variant")
METADATA = ("tokenizer.json", "run.json", "data-meta.json", "environment.txt", "metrics.jsonl")
SOURCE = ("model.py", "components.py", "train.py", "tools.py", "helpers/__init__.py",
          "helpers/bpe.py", "helpers/tokenizer.py", "helpers/data.py",
          "helpers/infrastructure.py", "helpers/tracking.py",
          "helpers/evaluate.py", "helpers/generate.py")
CHECKPOINT_KEYS = ("model", "model_config", "tokenizer", "eot_id", "source_hashes", "step", "data_hash")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def safe_path(name):
    path = PurePosixPath(name)
    if not name or "\\" in name or path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError(f"Unsafe package path: {name!r}")
    return path


def no_links(path):
    path = Path(os.path.abspath(path))
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"Symlinks are not accepted: {path.name}")
    return path


def source_path(name):
    path = safe_path(name)
    if path.suffix != ".py" or any(part.startswith(".") for part in path.parts):
        raise ValueError("Invalid saved source filename")
    return path


def read_regular(path):
    path = no_links(path)
    if not path.is_file():
        raise ValueError(f"Required regular file missing: {path.name}")
    return path.read_bytes()


def reject_obvious_secrets(data, name):
    # An additional guard, not a guarantee that arbitrary source contains no secrets.
    if re.search(rb"\b(?:hf_|ghp_)[A-Za-z0-9]{20,}\b|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----", data):
        raise ValueError(f"Possible credential in {name}; remove it before packaging")


def checkpoint_checks(folder):
    saved = torch.load(io.BytesIO(read_regular(folder / "best.pt")), map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or set(saved) != set(CHECKPOINT_KEYS):
        raise ValueError("Checkpoint fields differ from the required seven-field format")
    run = json.loads(read_regular(folder / "run.json"))
    meta = json.loads(read_regular(folder / "data-meta.json"))
    tokenizer = read_regular(folder / "tokenizer.json")
    if saved["tokenizer"].encode() != tokenizer:
        raise ValueError("Checkpoint and tokenizer.json differ")
    if saved["model_config"] != run["model"]:
        raise ValueError("Checkpoint and run.json model configurations differ")
    if saved["eot_id"] != meta["eot_id"] or saved["model_config"]["vocab_size"] != meta["vocab_size"]:
        raise ValueError("Checkpoint and data metadata vocabulary/end token differ")
    if digest(tokenizer) != meta["files"]["tokenizer.json"]:
        raise ValueError("Tokenizer does not match data metadata hash")
    if run.get("data_hash") != digest(read_regular(folder / "data-meta.json")):
        raise ValueError("Run data metadata hash differs")
    if saved["data_hash"] != run["data_hash"]:
        raise ValueError("Checkpoint data hash differs from run")
    hashes = saved["source_hashes"]
    if not hashes or hashes != run["source_hashes"]:
        raise ValueError("Checkpoint and run source fingerprints differ")
    actual = {path.relative_to(folder / "source").as_posix()
              for path in (folder / "source").rglob("*.py")
              if not any(part.startswith(".") for part in path.relative_to(folder / "source").parts)}
    if actual != set(hashes):
        raise ValueError("Saved source file list differs from checkpoint fingerprints")
    for name, expected in hashes.items():
        source_path(name)
        if digest(read_regular(folder / "source" / name)) != expected:
            raise ValueError(f"Saved source changed: {name}")
    for name in SOURCE:
        read_regular(folder / "source" / name)
    if not isinstance(saved["model"], dict) or not saved["model"]:
        raise ValueError("Empty or invalid model state")
    for value in saved["model"].values():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError("Model state must contain finite tensors")
    return saved


SMOKE = r'''
import json, pathlib, sys, torch
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root / 'source'))
from model import Config, GPT
from helpers.tokenizer import Tokenizer
torch.set_num_threads(2)
saved = torch.load(root / 'best.pt', map_location='cpu', weights_only=True)
tokenizer = Tokenizer.from_str(saved['tokenizer'])
assert tokenizer.get_vocab_size() == saved['model_config']['vocab_size'], 'Tokenizer size mismatch'
assert tokenizer.token_to_id('<|endoftext|>') == saved['eot_id'], 'End token mismatch'
model = GPT(Config(**saved['model_config'])).eval()
model.load_state_dict(saved['model'], strict=True)
ids = [saved['eot_id']] + tokenizer.encode('A').ids
with torch.no_grad():
    logits, _ = model(torch.tensor([ids]))
assert torch.isfinite(logits).all(), 'Nonfinite logits'
result = model.generate(ids, 2, temperature=0, top_k=0, stop=saved['eot_id'])
assert len(ids) <= len(result) <= len(ids) + 2
print(json.dumps({'loaded': True, 'generated_tokens': len(result) - len(ids)}))
'''


def smoke(folder):
    # Isolation avoids importing the other run's model. It is NOT a security sandbox.
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", SMOKE, str(folder.resolve())],
                               capture_output=True, text=True, timeout=120,
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"})
    if completed.returncode:
        raise ValueError(f"Saved-source CPU check failed for {folder.name}:\n{completed.stderr[-4000:]}")
    return json.loads(completed.stdout)


def inventory(folder):
    result = {}
    for path in sorted(folder.rglob("*")):
        no_links(path)
        if path.is_dir():
            continue
        raw = read_regular(path)
        name = path.relative_to(folder).as_posix()
        result[name] = {"sha256": digest(raw), "bytes": len(raw)}
    return result


def validate_manifest(manifest):
    if manifest.get("format") != FORMAT or set(manifest.get("runs", {})) != set(ROLES):
        raise ValueError("Invalid submission manifest")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Missing manifest files")
    for name, info in files.items():
        path = safe_path(name)
        valid = name == "README.md" or (
            len(path.parts) == 2 and path.parts[0] == "tests" and path.name.startswith("test_") and path.suffix == ".py"
        ) or (
            path.parts[0] in ROLES and (
                len(path.parts) == 2 and path.name in (*METADATA, "best.pt") or
                len(path.parts) >= 3 and path.parts[1] == "source" and path.suffix == ".py"
                and all(not part.startswith(".") for part in path.parts[2:])
            )
        )
        if not valid or not isinstance(info, dict) or not re.fullmatch(r"[0-9a-f]{64}", info.get("sha256", "")):
            raise ValueError(f"Unexpected file or invalid checksum: {name}")
        if type(info.get("bytes")) is not int or info["bytes"] < 0:
            raise ValueError(f"Invalid byte count: {name}")
    required = {"README.md"}
    for role in ROLES:
        required.update(f"{role}/{name}" for name in (*METADATA, "best.pt"))
        required.update(f"{role}/source/{name}" for name in SOURCE)
    if not required <= files.keys() or not any(name.startswith("tests/") for name in files):
        raise ValueError("Package lacks required run files or student tests")
    return manifest


def verify(folder, *, run_code=False):
    folder = no_links(folder)
    manifest = validate_manifest(json.loads(read_regular(folder / "manifest.json")))
    actual = inventory(folder)
    actual.pop("manifest.json", None)
    if actual != manifest["files"]:
        raise ValueError("Package file list/checksums differ from manifest")
    for role in ROLES:
        checkpoint_checks(folder / role)
        if run_code:
            smoke(folder / role)
    return manifest


def remove_stage(folder):
    if folder.exists():
        shutil.rmtree(folder)


def new_stage(out):
    out = no_links(out)
    if out.exists():
        raise FileExistsError("Choose a new output path; existing packages are never replaced")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out, Path(tempfile.mkdtemp(prefix=".submission-", dir=out.parent))


def package(baseline, variant, tests, out):
    runs = {"baseline": no_links(baseline), "variant": no_links(variant)}
    if runs["baseline"] == runs["variant"]:
        raise ValueError("Select two distinct run directories")
    tests = [no_links(path) for path in tests]
    if not tests or len({p.name for p in tests}) != len(tests):
        raise ValueError("Supply student test files with distinct filenames")
    for path in tests:
        if not path.name.startswith("test_") or path.suffix != ".py":
            raise ValueError("Student tests must be test_*.py files")
    out, stage = new_stage(out)
    try:
        run_info = {}
        for role, source in runs.items():
            saved = checkpoint_checks(source)
            destination = stage / role
            (destination / "source").mkdir(parents=True)
            for name in METADATA:
                (destination / name).write_bytes(read_regular(source / name))
            for name in sorted(saved["source_hashes"]):
                target = destination / "source" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(read_regular(source / "source" / name))
            checkpoint_bytes = read_regular(source / "best.pt")
            (destination / "best.pt").write_bytes(checkpoint_bytes)
            run_info[role] = {"original_checkpoint_sha256": digest(checkpoint_bytes),
                              "selected_step": saved["step"]}
        (stage / "tests").mkdir()
        for path in tests:
            (stage / "tests" / path.name).write_bytes(read_regular(path))
        (stage / "README.md").write_text(
            "# Course model submission\n\n"
            "Baseline and variant retain separate saved Python source. These are custom PyTorch models.\n"
            "Use the complete course environment, including PyTorch and tokenizer dependencies; environment.txt records the training packages.\n"
            "Only run source you trust. From this package directory:\n\n"
            "```sh\npython baseline/source/tools.py generate baseline/best.pt --device cpu --prompt A --tokens 2\n"
            "python variant/source/tools.py generate variant/best.pt --device cpu --prompt A --tokens 2\n"
            "python baseline/source/tools.py evaluate baseline/best.pt --text EVAL_TEXT --device cpu\n"
            "python variant/source/tools.py evaluate variant/best.pt --text EVAL_TEXT --device cpu\n```\n\n"
            "Evaluation text is supplied separately.\n"
            "Student tests are in tests/. Preserve manifest.json and the submitted Hub commit SHA.\n",
            encoding="utf-8")
        for path in stage.rglob("*"):
            if path.is_file() and path.suffix != ".pt":
                reject_obvious_secrets(path.read_bytes(), path.relative_to(stage).as_posix())
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                capture_output=True, text=True) if shutil.which("git") else None
        provenance = commit.stdout.strip() if commit and commit.returncode == 0 else None
        manifest = {"format": FORMAT, "runs": run_info, "files": inventory(stage),
                    "packaging_checkout_commit": provenance,
                    "provenance_note": "Checkout commit at packaging, not a claim about training; saved source hashes identify code."}
        (stage / "manifest.json").write_bytes(json_bytes(manifest))
        verify(stage, run_code=True)
        stage.rename(out)
        return {"package": str(out), "manifest_sha256": digest((out / "manifest.json").read_bytes()),
                "files": len(manifest["files"])}
    except BaseException:
        remove_stage(stage)
        raise


def hub_api():
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError("Install assignment1/code/requirements.txt in your course environment") from error
    return HfApi()


def upload(folder, repo_id, receipt):
    from huggingface_hub import CommitOperationAdd
    folder = no_links(folder)
    manifest = verify(folder)
    receipt = no_links(receipt)
    if receipt.exists() or folder == receipt or folder in receipt.parents:
        raise ValueError("Receipt must be a new file outside the package")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo_id):
        raise ValueError("Use an explicit USERNAME/NEW_REPOSITORY")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    api = hub_api()
    # Fail on an existing repo: no changing visibility or replacing prior submissions.
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=False)
    names = ["manifest.json", *sorted(manifest["files"])]
    result = api.create_commit(repo_id=repo_id, repo_type="model",
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(folder / name)) for name in names],
        commit_message="Submit baseline and variant with checksummed source")
    record = {"format": FORMAT, "repo_id": repo_id, "revision": result.oid,
              "commit_url": result.commit_url,
              "manifest_sha256": digest((folder / "manifest.json").read_bytes())}
    try:
        with receipt.open("x", encoding="utf-8") as stream:
            stream.write(json_bytes(record).decode())
    except OSError as error:
        raise RuntimeError(f"Upload succeeded at {result.commit_url}, revision {result.oid}, "
                           f"manifest SHA-256 {record['manifest_sha256']}, but receipt could not be saved") from error
    return record


def download_check(receipt, out, *, trust_code=False):
    if not trust_code:
        raise ValueError("Checking executes downloaded Python. Use --trust-code only for your own submission")
    record = json.loads(read_regular(receipt))
    revision = record.get("revision", "")
    if record.get("format") != FORMAT or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Receipt must pin a full 40-character commit SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", record.get("manifest_sha256", "")):
        raise ValueError("Receipt must contain the manifest SHA-256")
    from huggingface_hub import hf_hub_download
    out, stage = new_stage(out)
    try:
        def fetch(name):
            safe_path(name)
            cached = hf_hub_download(repo_id=record["repo_id"], repo_type="model", filename=name,
                                     revision=revision, token=False)
            # Hub cache can use symlinks. Materialize independent regular files.
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(cached).read_bytes())
        fetch("manifest.json")
        if digest((stage / "manifest.json").read_bytes()) != record["manifest_sha256"]:
            raise ValueError("Downloaded manifest differs from upload receipt")
        manifest = validate_manifest(json.loads((stage / "manifest.json").read_text()))
        for name in sorted(manifest["files"]):
            fetch(name)
        verify(stage, run_code=True)
        stage.rename(out)
        return {"checked": True, "repo_id": record["repo_id"], "revision": revision,
                "download": str(out), "cpu_checks": list(ROLES)}
    except BaseException:
        remove_stage(stage)
        raise


def main(argv=None):
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("package", help="Validate your own runs and create a package with checksums")
    pack.add_argument("--baseline", required=True, type=Path)
    pack.add_argument("--variant", required=True, type=Path)
    pack.add_argument("--tests", required=True, nargs="+", type=Path)
    pack.add_argument("--out", required=True, type=Path)
    publish = commands.add_parser("upload", help="PUBLICATION: create a NEW PUBLIC model repository")
    publish.add_argument("package", type=Path)
    publish.add_argument("--repo-id", required=True)
    publish.add_argument("--receipt", required=True, type=Path)
    check = commands.add_parser("check", help="Download exact receipt revision and run trusted CPU checks")
    check.add_argument("--receipt", required=True, type=Path)
    check.add_argument("--out", required=True, type=Path)
    check.add_argument("--trust-code", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "package":
        result = package(args.baseline, args.variant, args.tests, args.out)
    elif args.command == "upload":
        result = upload(args.package, args.repo_id, args.receipt)
    else:
        result = download_check(args.receipt, args.out, trust_code=args.trust_code)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
