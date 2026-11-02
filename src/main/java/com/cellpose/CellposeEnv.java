package com.cellpose;

import java.util.*;
import org.apposed.appose.Appose;
import org.apposed.appose.Environment;
import org.apposed.appose.Service;

/**
 * Shared Cellpose/Appose environment builder, with real GPU support.
 *
 * <p>Built from a generated {@code pixi.toml} (fed to Appose via
 * {@link org.apposed.appose.builder.PixiBuilder#content}). This is required to get a
 * CUDA-enabled PyTorch: pixi installs the CPU-only torch build unless (a) the target
 * platform declares CUDA via {@code { platform = "win-64", cuda = "12.4" }}, and
 * (b) torch/torchvision are pinned to the PyTorch CUDA wheel index. Both are done below,
 * following pixi's official PyTorch guide
 * (https://prefix-dev.github.io/pixi/latest/python/pytorch/).
 *
 * <p>Platform behavior (each user builds the env on their own machine, so only their
 * platform is resolved):
 * <ul>
 *   <li><b>win-64 / linux-64 with an NVIDIA driver</b> &rarr; the {@code *-cuda} platform
 *       wins (listed first), installing torch from the {@code cu124} index. CUDA is
 *       backward compatible, so a cu124 build runs on any driver supporting CUDA 12.4+.</li>
 *   <li><b>win-64 / linux-64 without CUDA</b> &rarr; falls back to the {@code *-cpu}
 *       platform (torch from the cpu index).</li>
 *   <li><b>osx-arm64 (Apple Silicon)</b> &rarr; default macOS torch wheel, which includes
 *       MPS acceleration. No CUDA on macOS.</li>
 * </ul>
 *
 * <p>The device actually used is chosen at runtime in each worker (cuda &rarr; mps &rarr;
 * cpu). PYTHONUNBUFFERED=1 and the workers' {@code .init(...)} imports (see the command
 * classes) are still required to avoid the Appose Windows worker hang; those are
 * independent of this GPU work.
 *
 * <p>First build downloads gigabytes (CUDA torch on Windows/Linux) and may take several
 * minutes; the env is cached afterward under the name below.
 */
public final class CellposeEnv {

    private CellposeEnv() {}

    /**
     * A single pixi.toml covering all platforms. Torch/torchvision are pinned to the
     * CUDA wheel index on Windows/Linux and the CPU (MPS on Mac) wheel elsewhere;
     * cellpose and appose install from normal PyPI.
     *
     * <p>NB on syntax: the pixi bundled by Appose (~0.39) does NOT accept the newer
     * inline-table {@code platforms = [{ platform = "win-64", cuda = "12.4" }, ...]}
     * form (it errors "expected a string, found table"). CUDA availability must instead
     * be declared via the classic {@code [system-requirements]} table with plain-string
     * platforms. Without a CUDA declaration, pixi installs CPU-only torch.
     */
    private static final String PIXI_TOML = """
[workspace]
name = "cellpose-fiji"
channels = ["https://prefix.dev/conda-forge"]
platforms = ["win-64", "linux-64", "osx-arm64"]

[system-requirements]
# Declares CUDA 12 available to the solver so torch resolves its CUDA build on
# Windows/Linux. Harmless on macOS (its torch dep below uses no CUDA index).
cuda = "12"

[dependencies]
python = ">=3.11,<3.13"

[pypi-dependencies]
# cellpose (+ deps incl. numpy) and appose from normal PyPI.
# packaging is needed by fastremap (a cellpose dep) at import time but is not
# always pulled in transitively, so list it explicitly.
cellpose = "==4.2.1.1"
appose = "*"
packaging = "*"
# Windows/Linux: CUDA 12.4 torch wheels.
torch = { version = ">=2.6", index = "https://download.pytorch.org/whl/cu124" }
torchvision = { version = "*", index = "https://download.pytorch.org/whl/cu124" }

[target.osx-arm64.pypi-dependencies]
# macOS (Apple Silicon): default wheel includes MPS; no CUDA on macOS.
torch = ">=2.6"
torchvision = "*"
""";

    // .init imports numpy/torch/cellpose at worker startup. Appose warns that
    // importing numpy in a task on Windows hangs (appose #23); the init script is
    // the sanctioned fix. Must be valid Python with real newlines (a text block).
    private static final String INIT_SCRIPT = """
import numpy
import torch
try:
    torch.cuda.is_available()
except Exception:
    pass
from cellpose import models
""";

    // --- Python: emits stage updates via task.update(...) ---
    // Device selection respects the "Use GPU" checkbox. numpy/torch/cellpose are
    // imported in the .init script at worker startup — importing them in-task hangs
    // the worker on Windows (Appose #23), so they must be pre-imported.
    private static final String WORKER = """
import numpy as np
import torch
from cellpose import models

task.update("Loading model", current=1, maximum=3)

cuda_ok = torch.cuda.is_available()
mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

if use_gpu and cuda_ok:
    device = torch.device("cuda")
elif use_gpu and mps_ok:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

img = image.ndarray()

if img.shape[0] == 1:
    img = img[0]
    ch_axis = None
else:
    ch_axis = 0

if cell_diam == 0:
    dia = None
else:
    dia = cell_diam

gpu = device.type != "cpu"
model = models.CellposeModel(gpu=gpu, device=device, pretrained_model=custom_model)

task.update("Segmenting (" + device.type + ")", current=2, maximum=3)
masks, flows, styles = model.eval(img, channel_axis=ch_axis, flow_threshold=flow_thresh, diameter=dia, min_size=min_cell_size, cellprob_threshold=cell_prob)

task.update("Writing masks", current=3, maximum=3)
res = mask.ndarray()
res[:] = masks.astype("int32")
""";

    public static Environment build() throws Exception {
        return Appose.pixi()
            .content(PIXI_TOML)
            .env("PYTHONUNBUFFERED", "1")
            .name("cellpose-fiji")
            .build();
    }

    public static Service init() throws Exception {
        return build().python().init(INIT_SCRIPT);
    }

    public static Service.Task run(Service python, Map<String, Object> inputs) {
        return python.task(WORKER, inputs);
    }
}
