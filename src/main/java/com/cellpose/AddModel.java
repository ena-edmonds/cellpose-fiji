package com.cellpose;

import org.apposed.appose.Environment;
import org.apposed.appose.Service;

import ij.IJ;

import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.plugin.Menu;
import org.scijava.menu.MenuConstants;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

/**
 * Add a Cellpose model to the gui model list, ported from the add_model script.
 * <p>The trailing {@code @Menu} weight (3) places it below Segment Image (1) and
 * Batch Segment Folder (2).
 */
@Plugin(type = Command.class, menu = {
    @Menu(label = MenuConstants.ANALYZE_LABEL,
          weight = MenuConstants.ANALYZE_WEIGHT,
          mnemonic = MenuConstants.ANALYZE_MNEMONIC),
    @Menu(label = "Cellpose", weight = 100, mnemonic = 'c'),
    @Menu(label = "Add Model", weight = 3, mnemonic = 'a')
})
public class AddModel implements Command {

    // Take in a file as input, convert that into a string of the path to it
    @Parameter(label = "Select a cellpose model to add to the list", style = "file")
    private File newModel;

    @Override
    public void run() {
        String filePath = newModel.getAbsolutePath();

        try {
            // Build the Cellpose environment (slow only on first run of any part of the plugin)
            IJ.showStatus("Cellpose: preparing environment (first run may take minutes)…");
            IJ.showProgress(0.15);
            Environment env = CellposeEnv.build();

            String script = """
import os
from cellpose import io, models

# On a fresh machine ~/.cellpose/models/ may not exist yet; add_model() copies
# into models.MODEL_DIR without creating it, so ensure it exists first.
os.makedirs(models.MODEL_DIR, exist_ok=True)
io.add_model(new_model)
""";

            IJ.showStatus("Cellpose: adding model…");
            IJ.showProgress(0.70);

            // Windows: PYTHONUNBUFFERED=1 (set in CellposeEnv) makes the worker flush its
            // COMPLETION response instead of block-buffering it in the stdout pipe, and
            // .init(...) imports cellpose at worker startup rather than in-task. No torch
            // here: torch.cuda.is_available() hangs in the Appose worker on this platform,
            // so Add Model deliberately does not probe the GPU.
            try (Service python = env.python().init("from cellpose import io, models")) {
                Map<String, Object> inputs = new HashMap<>();
                inputs.put("new_model", filePath);
                Service.Task task = python.task(script, inputs);
                task.waitFor();
            }

            IJ.showStatus("Cellpose: model succesfully added!");
            IJ.showProgress(1.0);
        } catch (Exception e) {
            IJ.showProgress(1.0);
            IJ.handleException(e);
        }
    }
}
