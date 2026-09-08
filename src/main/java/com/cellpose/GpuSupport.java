package com.cellpose;

import ij.Prefs;

/**
 * Remembers, across runs, whether this machine can use a GPU for Cellpose, and
 * what the user last chose for the "Use GPU" checkbox. Both are stored in ImageJ
 * {@link Prefs} (persisted in {@code IJ_Prefs.txt}).
 *
 * <p>Why cache availability: the only authoritative GPU check is PyTorch
 * ({@code torch.cuda.is_available()} / {@code torch.backends.mps.is_available()}),
 * which can only run inside the built Appose environment — after the "Use GPU"
 * checkbox has already been drawn. So the checkbox uses the <em>cached</em> result
 * from a previous run; the current run refreshes the cache once it has actually
 * talked to PyTorch (see {@link #recordAvailability(boolean)}).
 *
 * <p>First-run behavior: availability is unknown, so the checkbox is shown
 * (defaulting off). After that run reports back, subsequent runs hide the box on
 * machines with no usable GPU, and show it defaulting on where a GPU exists.
 *
 * <p>Resetting: if a user adds a GPU or fixes drivers later, the cached "no GPU"
 * would otherwise stick. Delete the {@code cellpose.gpuAvailable} entry from
 * {@code IJ_Prefs.txt} (or call {@link #clearAvailability()}) to force a re-probe.
 */
public final class GpuSupport {

    /** Cached GPU availability: "true", "false", or absent (unknown). */
    private static final String KEY_AVAILABLE = "cellpose.gpuAvailable";
    /** Remembered last checkbox choice (global across platforms). */
    private static final String KEY_LAST_CHOICE = "cellpose.useGpu";

    private GpuSupport() {}

    /** Availability as a tri-state: TRUE, FALSE, or null when never probed. */
    public static Boolean availability() {
        String v = Prefs.get(KEY_AVAILABLE, null);
        if (v == null) return null;
        return Boolean.valueOf(v);
    }

    /** True only when we have positively determined a GPU is usable. */
    public static boolean isKnownAvailable() {
        return Boolean.TRUE.equals(availability());
    }

    /** True only when we have positively determined no GPU is usable. */
    public static boolean isKnownUnavailable() {
        return Boolean.FALSE.equals(availability());
    }

    /** Persist a freshly observed availability result (from PyTorch). */
    public static void recordAvailability(boolean available) {
        Prefs.set(KEY_AVAILABLE, Boolean.toString(available));
        Prefs.savePreferences();
    }

    /** Forget cached availability, forcing the next run to re-probe/show the box. */
    public static void clearAvailability() {
        Prefs.set(KEY_AVAILABLE, null);
        Prefs.savePreferences();
    }

    /**
     * Whether the "Use GPU" checkbox should appear at all.
     * Hidden only when we positively know there is no usable GPU.
     */
    public static boolean shouldShowCheckbox() {
        return !isKnownUnavailable();
    }

    /**
     * Default state for the checkbox when it is shown.
     * <ul>
     *   <li>If the user has a remembered choice, use it.</li>
     *   <li>Else default on when a GPU is known available, off otherwise
     *       (including the unknown first-run case).</li>
     * </ul>
     */
    public static boolean defaultChecked() {
        String last = Prefs.get(KEY_LAST_CHOICE, null);
        if (last != null) return Boolean.parseBoolean(last);
        return isKnownAvailable();
    }

    /** Remember the user's checkbox choice for next time. */
    public static void recordChoice(boolean checked) {
        Prefs.set(KEY_LAST_CHOICE, Boolean.toString(checked));
        Prefs.savePreferences();
    }
}
