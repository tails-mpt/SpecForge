import argparse
from dataclasses import dataclass
from typing import Any, Dict, List

from sglang.srt.server_args import ATTENTION_BACKEND_CHOICES


@dataclass
class TrackerArgs:
    report_to: str = "none"
    wandb_project: str = None
    wandb_name: str = None
    wandb_key: str = None
    wandb_offline: bool = False
    wandb_dir: str = None
    swanlab_project: str = None
    swanlab_name: str = None
    swanlab_key: str = None
    mlflow_experiment_id: str = None
    mlflow_run_name: str = None
    mlflow_run_id: str = None
    mlflow_tracking_uri: str = None
    mlflow_registry_uri: str = None

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--report-to",
            type=str,
            default="none",
            choices=["wandb", "tensorboard", "swanlab", "mlflow", "none"],
            help="The integration to report results and logs to.",
        )
        # wandb-specific args
        parser.add_argument("--wandb-project", type=str, default=None)
        parser.add_argument("--wandb-name", type=str, default=None)
        parser.add_argument("--wandb-key", type=str, default=None, help="W&B API key.")
        parser.add_argument(
            "--wandb-offline",
            action="store_true",
            help="Enable W&B offline mode and store logs locally.",
        )
        parser.add_argument(
            "--wandb-dir",
            type=str,
            default=None,
            help="Directory to store W&B files. Defaults to './wandb' under the project root when using W&B.",
        )
        # swanlab-specific args
        parser.add_argument(
            "--swanlab-project",
            type=str,
            default=None,
            help="The project name for swanlab.",
        )
        parser.add_argument(
            "--swanlab-name",
            type=str,
            default=None,
            help="The experiment name for swanlab.",
        )
        parser.add_argument(
            "--swanlab-key",
            type=str,
            default=None,
            help="The API key for swanlab non-interactive login.",
        )
        # mlflow-specific args
        parser.add_argument(
            "--mlflow-tracking-uri",
            type=str,
            default=None,
            help="The MLflow tracking URI. If not set, uses MLFLOW_TRACKING_URI environment variable or defaults to local './mlruns'.",
        )
        parser.add_argument(
            "--mlflow-experiment-name",
            type=str,
            default=None,
            help="The MLflow experiment name. If not set, uses MLFLOW_EXPERIMENT_NAME environment variable.",
        )
        parser.add_argument(
            "--mlflow-run-name",
            type=str,
            default=None,
            help="The MLflow run name. If not set, MLflow will auto-generate one.",
        )


@dataclass
class SGLangBackendArgs:
    sglang_attention_backend: str = "fa3"
    sglang_mem_fraction_static: float = 0.4
    sglang_context_length: int = None
    sglang_enable_nccl_nvls: bool = False
    sglang_enable_symm_mem: bool = False
    sglang_enable_torch_compile: bool = True
    sglang_enable_dp_attention: bool = False
    sglang_enable_dp_lm_head: bool = False
    sglang_enable_piecewise_cuda_graph: bool = False
    sglang_piecewise_cuda_graph_max_tokens: int = 4096
    sglang_piecewise_cuda_graph_tokens: List[int] = None
    sglang_ep_size: int = 1
    sglang_max_running_requests: int = None  # assign based on batch size
    sglang_max_total_tokens: int = None  # assign based on batch size and seq length
    # Crucible Squeeze (in-VM 2026-05-03): expose sglang's --fp8-gemm-backend and
    # --moe-runner-backend so we can force triton on FP8 MoE targets where the
    # auto path picks DeepGEMM and the JIT compile crashes (kernel_runtime.hpp:45
    # exit_code == 0). SpecForge's training path goes through SGLangRunner (NOT
    # the sglang Scheduler), so initialize_fp8_gemm_config never runs and the
    # --fp8-gemm-backend flag at sglang.launch_server level is also bypassed.
    # The fix is to forward these directly into ServerArgs(...) construction
    # via to_kwargs() below.
    sglang_fp8_gemm_backend: str = "auto"
    sglang_moe_runner_backend: str = "auto"
    # Crucible nemotron-fp32-extract (in-VM 2026-07-21, branch
    # nemotron-fp32-extract-h200): forward sglang's --mamba-ssm-dtype into the
    # in-process teacher-extraction engine so the Mamba-2 SSM prefill state is
    # kept in fp32 (higher-precision teacher). Default "float32" to test a
    # suspected degraded-teacher root cause for a low EAGLE-3 acc_0 ceiling on
    # the hybrid nemotron_h target. Harmless no-op for dense targets: they have
    # no Mamba layers, so ServerArgs only sets the SGLANG_MAMBA_SSM_DTYPE env
    # var (which is never read) and, with sglang's default linear_attn_backend
    # ="triton"/linear_attn_decode_backend=None, the SM100+ flashinfer-GDN
    # bf16-required check never triggers. Pass "bfloat16" to restore the
    # previous (implicit-bf16) behavior.
    sglang_mamba_ssm_dtype: str = "float32"

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        # sglang arguments
        parser.add_argument(
            "--sglang-attention-backend",
            type=str,
            default="flashinfer",
            choices=ATTENTION_BACKEND_CHOICES,
            help="The attention backend of SGLang backend",
        )
        parser.add_argument(
            "--sglang-mem-fraction-static",
            type=float,
            default=0.4,
            help="The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors.",
        )
        parser.add_argument(
            "--sglang-context-length",
            type=int,
            default=None,
            help="The context length of the SGLang backend",
        )
        parser.add_argument(
            "--sglang-enable-nccl-nvls",
            action="store_true",
            help="Enable NCCL NVLS for prefill heavy requests when available for SGLang backend",
        )
        parser.add_argument(
            "--sglang-enable-symm-mem",
            action="store_true",
            help="Enable NCCL symmetric memory for fast collectives for SGLang backend",
        )
        parser.add_argument(
            "--sglang-enable-torch-compile",
            action="store_true",
            help="Optimize the model with torch.compile for SGLang backend",
        )
        parser.add_argument(
            "--sglang-enable-dp-attention",
            action="store_true",
            help="Enable DP attention for SGLang backend",
        )
        parser.add_argument(
            "--sglang-enable-dp-lm-head",
            action="store_true",
            help="Enable piecewise CUDA graph for SGLang backend",
        )
        parser.add_argument(
            "--sglang-enable-piecewise-cuda-graph",
            action="store_true",
            help="Enable piecewise CUDA graph for SGLang backend's prefill",
        )
        parser.add_argument(
            "--sglang-piecewise-cuda-graph-max-tokens",
            type=int,
            default=4096,
            help="Set the max tokens for piecewise CUDA graph for SGLang backend",
        )
        parser.add_argument(
            "--sglang-piecewise-cuda-graph-tokens",
            type=int,
            nargs="+",
            default=None,
            help="Set the list of tokens when using piecewise cuda graph for SGLang backend",
        )
        parser.add_argument(
            "--sglang-ep-size",
            type=int,
            default=1,
            help="The ep size of the SGLang backend",
        )
        parser.add_argument(
            "--sglang-fp8-gemm-backend",
            type=str,
            default="auto",
            help=(
                "Forwarded to sglang's ServerArgs.fp8_gemm_runner_backend (the "
                "--fp8-gemm-backend flag on sglang.launch_server). "
                "'auto' lets sglang pick (DeepGEMM on H100 sm90 — which JIT-crashes "
                "on some venvs); pass 'triton' to force the triton path which is "
                "more robust at the cost of ~10-20%% kernel time."
            ),
        )
        parser.add_argument(
            "--sglang-moe-runner-backend",
            type=str,
            default="auto",
            help=(
                "Forwarded to sglang's ServerArgs.moe_runner_backend. Same intent "
                "as --sglang-fp8-gemm-backend but for the MoE expert-forward path."
            ),
        )
        parser.add_argument(
            "--sglang-mamba-ssm-dtype",
            type=str,
            default="float32",
            choices=["float32", "bfloat16", "float16"],
            help=(
                "Forwarded to sglang's ServerArgs.mamba_ssm_dtype (the "
                "--mamba-ssm-dtype flag). Controls the dtype of the Mamba-2 SSM "
                "state during teacher hidden-state/logit extraction. Default "
                "'float32' keeps the hybrid (nemotron_h) teacher in high precision; "
                "harmless no-op for dense targets. Pass 'bfloat16' for the previous "
                "implicit behavior."
            ),
        )

    @staticmethod
    def from_args(args: argparse.Namespace) -> "SGLangBackendArgs":
        return SGLangBackendArgs(
            sglang_attention_backend=args.sglang_attention_backend,
            sglang_mem_fraction_static=args.sglang_mem_fraction_static,
            sglang_context_length=args.sglang_context_length,
            sglang_enable_nccl_nvls=args.sglang_enable_nccl_nvls,
            sglang_enable_symm_mem=args.sglang_enable_symm_mem,
            sglang_enable_torch_compile=args.sglang_enable_torch_compile,
            sglang_enable_dp_attention=args.sglang_enable_dp_attention,
            sglang_enable_dp_lm_head=args.sglang_enable_dp_lm_head,
            sglang_enable_piecewise_cuda_graph=args.sglang_enable_piecewise_cuda_graph,
            sglang_piecewise_cuda_graph_max_tokens=args.sglang_piecewise_cuda_graph_max_tokens,
            sglang_piecewise_cuda_graph_tokens=args.sglang_piecewise_cuda_graph_tokens,
            sglang_ep_size=args.sglang_ep_size,
            sglang_max_running_requests=(
                args.target_batch_size if hasattr(args, "target_batch_size") else None
            ),
            sglang_max_total_tokens=(
                args.target_batch_size * args.max_length * getattr(args, "ttt_length", 7)
                if hasattr(args, "target_batch_size") and hasattr(args, "max_length")
                else None
            ),
            sglang_fp8_gemm_backend=getattr(args, "sglang_fp8_gemm_backend", "auto"),
            sglang_moe_runner_backend=getattr(args, "sglang_moe_runner_backend", "auto"),
            sglang_mamba_ssm_dtype=getattr(
                args, "sglang_mamba_ssm_dtype", "float32"
            ),
        )

    def to_kwargs(self) -> Dict[str, Any]:
        return dict(
            attention_backend=self.sglang_attention_backend,
            mem_fraction_static=self.sglang_mem_fraction_static,
            context_length=self.sglang_context_length,
            enable_nccl_nvls=self.sglang_enable_nccl_nvls,
            enable_symm_mem=self.sglang_enable_symm_mem,
            enable_torch_compile=self.sglang_enable_torch_compile,
            enable_dp_attention=self.sglang_enable_dp_attention,
            enable_dp_lm_head=self.sglang_enable_dp_lm_head,
            disable_piecewise_cuda_graph=not self.sglang_enable_piecewise_cuda_graph,
            piecewise_cuda_graph_max_tokens=self.sglang_piecewise_cuda_graph_max_tokens,
            piecewise_cuda_graph_tokens=self.sglang_piecewise_cuda_graph_tokens,
            ep_size=self.sglang_ep_size,
            max_running_requests=self.sglang_max_running_requests,
            max_total_tokens=self.sglang_max_total_tokens,
            # See class docstring above: forwarded to sglang's ServerArgs as
            # `fp8_gemm_runner_backend` / `moe_runner_backend`.
            fp8_gemm_runner_backend=self.sglang_fp8_gemm_backend,
            moe_runner_backend=self.sglang_moe_runner_backend,
            # Forwarded to sglang's ServerArgs.mamba_ssm_dtype; keeps the Mamba-2
            # SSM prefill state in fp32 for the hybrid teacher (no-op for dense).
            mamba_ssm_dtype=self.sglang_mamba_ssm_dtype,
        )
