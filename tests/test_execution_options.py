from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import BytesIO, StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts import train
from src.models.encoder import VisionTransformer
from src.models.predictor import VisionTransformerPredictor
from src.optimizer import create_optimizer
from src.train_step import train_step


class TestExecutionOptions(unittest.TestCase):
    def test_defaults_json_types_and_cli_overrides(self):
        options = ("compile", "fused_adamw", "cudnn_benchmark")
        args = train.parse_args(["--smoke-test"])[1]
        self.assertTrue(all(getattr(args, key) is False for key in options))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            config = dict(smoke_test=True, device="cuda", **dict.fromkeys(options, True))
            path.write_text(json.dumps(config))
            args = train.parse_args(["--config", str(path)])[1]
            self.assertTrue(all(getattr(args, key) is True for key in options))
            args = train.parse_args(["--config", str(path), "--no-compile",
                                     "--no-fused-adamw", "--no-cudnn-benchmark", "--device", "cpu"])[1]
            self.assertTrue(all(getattr(args, key) is False for key in options))
            for key in options:
                for value in (1, "false", None):
                    with self.subTest(key=key, value=value):
                        path.write_text(json.dumps({**config, key: value}))
                        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                            train.parse_args(["--config", str(path)])

    def test_cuda_only_options_reject_other_devices(self):
        for device in ("cpu", "mps"):
            for flag in ("--fused-adamw", "--cudnn-benchmark"):
                with self.subTest(device=device, flag=flag):
                    error = StringIO()
                    with redirect_stderr(error), self.assertRaises(SystemExit):
                        train.parse_args(["--smoke-test", "--device", device, flag])
                    self.assertIn("require a CUDA device", error.getvalue())
        self.assertTrue(train.parse_args(["--smoke-test", "--compile"])[1].compile)

    def test_remote_configs_enable_execution_options(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("sigreg", "visreg"):
            args = train.parse_args(["--config", str(root / "configs" /
                                    f"imagenet100_vit_base_400ep_sk_container_{name}.json")])[1]
            self.assertTrue(args.compile and args.fused_adamw and args.cudnn_benchmark)
            self.assertEqual((args.precision, args.batch_size), ("bf16", 128))

    def test_resume_ignores_only_optimizer_execution_defaults(self):
        config = {"optimizer": {"name": "AdamW", "defaults": {
            "lr": 0.001, "betas": [0.9, 0.999], "fused": None, "foreach": None,
        }}}
        other = deepcopy(config)
        other["optimizer"]["defaults"].update(fused=True, foreach=False)
        self.assertEqual(train.resume_config(config), train.resume_config(other))
        self.assertIsNone(config["optimizer"]["defaults"]["fused"])
        for key, value in (("lr", 0.002), ("betas", [0.8, 0.99]), ("eps", 1e-7)):
            changed = deepcopy(other)
            changed["optimizer"]["defaults"][key] = value
            self.assertNotEqual(train.resume_config(config), train.resume_config(changed))

    def test_launcher_resumes_with_compilation_on_and_off(self):
        original_compile = torch.compile
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            checkpoint_path = output / "checkpoint.pt"
            config_path = output / "config.json"
            keys = None
            for step, flag in ((1, "--no-compile"), (2, "--compile"), (3, "--no-compile")):
                arguments = ["train.py", "--smoke-test", "--mode", "joint", "--regularizer", "visreg",
                             "--output", str(output), "--max-steps", str(step), flag]
                if step > 1:
                    checkpoint = torch.load(checkpoint_path, weights_only=True)
                    config = json.loads(config_path.read_text())
                    for settings in (checkpoint["config"], config):
                        settings["optimizer"]["defaults"].update(fused=True, foreach=False)
                        settings.update(fused_adamw=True, cudnn_benchmark=True)
                    for group in checkpoint["optimizer"]["param_groups"]:
                        group.update(fused=True, foreach=False)
                    torch.save(checkpoint, checkpoint_path)
                    config_path.write_text(json.dumps(config))
                    arguments.extend(("--resume", str(checkpoint_path)))
                with patch.object(sys, "argv", arguments), patch.dict(
                    "os.environ", {"COMET_API_KEY": "test-key"},
                ), patch.object(train, "start"), patch.object(train, "log_model"), patch(
                    "torch.compile", side_effect=lambda model, **kwargs:
                    original_compile(model, backend="aot_eager", **kwargs),
                ) as compile_call, redirect_stdout(StringIO()):
                    train.main()
                self.assertEqual(compile_call.call_count, 2 if step == 2 else 0)
                for call in compile_call.call_args_list:
                    self.assertEqual(call.kwargs, {"dynamic": True, "mode": "default"})
                checkpoint = torch.load(checkpoint_path, weights_only=True)
                current_keys = {name: set(checkpoint[name]) for name in ("encoder", "predictor")}
                if keys is not None:
                    self.assertEqual(current_keys, keys)
                keys = current_keys
                self.assertEqual(checkpoint["step"], step)
                self.assertEqual(checkpoint["config"]["compile"], step == 2)
                self.assertFalse(checkpoint["config"]["fused_adamw"])
                self.assertFalse(checkpoint["config"]["cudnn_benchmark"])
                for group in checkpoint["optimizer"]["param_groups"]:
                    self.assertIsNone(group["fused"])
                    self.assertIsNone(group["foreach"])
                self.assertTrue(all(state["step"].item() == step
                                    for state in checkpoint["optimizer"]["state"].values()))


class TestCompiledModels(unittest.TestCase):
    def test_cpu_aot_dynamic_outputs_gradients_training_and_checkpoint_keys(self):
        self.check_models("cpu", "aot_eager")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for default compiler parity")
    def test_cuda_dynamic_outputs_gradients_training_and_checkpoint_keys(self):
        self.check_models("cuda", "inductor")

    def check_models(self, device, backend):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        encoder = VisionTransformer(image_size=8, patch_size=4, embed_dim=12, depth=1, num_heads=3).to(device)
        predictor = VisionTransformerPredictor(4, embed_dim=12, predictor_embed_dim=8, depth=1, num_heads=2).to(device)
        compiled_encoder, compiled_predictor = deepcopy(encoder), deepcopy(predictor)
        compiled_encoder.compile(dynamic=True, mode="default", backend=backend)
        compiled_predictor.compile(dynamic=True, mode="default", backend=backend)
        images = torch.randn(2, 3, 8, 8, device=device)
        for context_length, target_length in ((2, 2), (3, 1), (1, 3)):
            context = torch.arange(context_length, device=device).expand(2, -1)
            targets = torch.arange(4 - target_length, 4, device=device).expand(2, -1)
            outputs = []
            for enc, pred in ((encoder, predictor), (compiled_encoder, compiled_predictor)):
                enc.zero_grad(set_to_none=True)
                pred.zero_grad(set_to_none=True)
                features = enc(images)
                prediction = pred(enc(images, context), context, [targets])
                (features.square().mean() + prediction[0].square().mean()).backward()
                outputs.append((features, prediction[0]))
            for eager, compiled in zip(*outputs):
                torch.testing.assert_close(eager, compiled, rtol=1e-4, atol=1e-6)
            for eager_module, compiled_module in ((encoder, compiled_encoder), (predictor, compiled_predictor)):
                self.assertEqual(list(eager_module.state_dict()), list(compiled_module.state_dict()))
                for eager, compiled in zip(eager_module.parameters(), compiled_module.parameters()):
                    if eager.requires_grad:
                        torch.testing.assert_close(eager.grad, compiled.grad, rtol=1e-3, atol=1e-6)

        for regularizer in ("sigreg", "visreg"):
            optimizers = [create_optimizer(enc, pred, lr=0.001, weight_decay=0.04)
                          for enc, pred in ((encoder, predictor), (compiled_encoder, compiled_predictor))]
            for length in (2, 3):
                context = torch.arange(length, device=device).expand(2, -1)
                targets = [torch.arange(4 - length, 4, device=device).expand(2, -1)]
                metrics = []
                for enc, pred, optimizer in ((encoder, predictor, optimizers[0]),
                                              (compiled_encoder, compiled_predictor, optimizers[1])):
                    torch.manual_seed(42)
                    metrics.append(train_step(enc, pred, None, optimizer, images, context, targets, None,
                                              mode="joint", regularizer=regularizer))
                for key in metrics[0]:
                    torch.testing.assert_close(metrics[0][key], metrics[1][key], rtol=1e-3, atol=1e-5)
                for eager, compiled in zip(encoder.parameters(), compiled_encoder.parameters()):
                    torch.testing.assert_close(eager, compiled, rtol=1e-3, atol=1e-5)
                for eager, compiled in zip(predictor.parameters(), compiled_predictor.parameters()):
                    torch.testing.assert_close(eager, compiled, rtol=1e-3, atol=1e-5)

        for source, destination in ((encoder, compiled_encoder), (compiled_encoder, encoder),
                                    (predictor, compiled_predictor), (compiled_predictor, predictor)):
            buffer = BytesIO()
            torch.save(source.state_dict(), buffer)
            buffer.seek(0)
            destination.load_state_dict(torch.load(buffer, weights_only=True))


if __name__ == "__main__":
    unittest.main()
