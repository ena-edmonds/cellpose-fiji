package com.cellpose;

import org.apposed.appose.Appose;
import org.apposed.appose.Environment;

import java.util.Locale;

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

    private static boolean isMac() {
        return System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("mac");
    }

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

    public static Environment build() throws Exception {
        return Appose.pixi()
            .content(PIXI_TOML)
            // PYTHONUNBUFFERED=1: on Windows the worker's stdout is block-buffered as a
            // pipe; without this the COMPLETION response never flushes and task.waitFor()
            // hangs even though the script finished.
            .env("PYTHONUNBUFFERED", "1")
            .name("cellpose-fiji")
            .build();
    }
}
