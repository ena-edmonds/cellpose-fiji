package com.cellpose;

import org.apposed.appose.Environment;
import org.apposed.appose.Service;

import ij.IJ;
import ij.gui.GenericDialog;
import ij.gui.YesNoCancelDialog;

import org.scijava.command.Command;
import org.scijava.plugin.Plugin;
import org.scijava.plugin.Menu;
import org.scijava.menu.MenuConstants;

import java.awt.Color;
import java.awt.Font;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Remove a Cellpose model, ported from the remove_model script.
 * <p>weight = 10 leaves a gap after Add Model (weight 3), so the modern (Swing)
 * UI auto-inserts a separator before this destructive action.
 */
@Plugin(type = Command.class, menu = {
    @Menu(label = MenuConstants.ANALYZE_LABEL,
          weight = MenuConstants.ANALYZE_WEIGHT,
          mnemonic = MenuConstants.ANALYZE_MNEMONIC),
    @Menu(label = "Cellpose", weight = 100, mnemonic = 'c'),
    @Menu(label = "Remove Model", weight = 10, mnemonic = 'r')
})
public class RemoveModel implements Command {

    @Override
    public void run() {
        try {
            // select from dropdown of added models which to remove
            String home = System.getProperty("user.home");
            File registry = new File(home, ".cellpose/models/gui_models.txt");
            List<String> guiModels = new ArrayList<>();
            // Registry may not exist on a fresh machine; treat missing as empty.
            if (registry.exists()) {
                for (String line : java.nio.file.Files.readAllLines(registry.toPath())) {
                    String trimmed = line.trim();
                    if (!trimmed.isEmpty()) guiModels.add(trimmed); // drop blank lines
                }
            }
            if (guiModels.isEmpty()) {
                IJ.showMessage("Remove Model", "No custom Cellpose models have been added yet.");
                return;
            }
            String[] models = guiModels.toArray(new String[0]);

            GenericDialog gd = new GenericDialog("Select model to remove");
            gd.addChoice("Model", models, models[0]); // label, options, default
            gd.addMessage(
                "Warning: Removing a model will also delete the file associated with it to save space!\n" +
                "Only remove models you are prepared to delete.",
                new Font("SansSerif", Font.PLAIN, 12), Color.RED);
            gd.showDialog();
            if (gd.wasCanceled()) return;
            String modelName = gd.getNextChoice();

            // confirmation that you want to delete the model
            YesNoCancelDialog confirm = new YesNoCancelDialog(null,
                    "Confirm deletion",
                    "Are you sure you want to permanently delete:\n\n" + modelName +
                    "\n\nThis cannot be undone.");
            if (!confirm.yesPressed()) return;   // treats No and Cancel as "abort"

            // build the Cellpose environment (slow only on first run of any part of the plugin)
            IJ.showStatus("Cellpose: preparing environment (first run may take minutes)…");
            IJ.showProgress(0.15);
            Environment env = CellposeEnv.build();

            String script = """
from cellpose import io
io.remove_model(model_name, delete=True)
""";

            IJ.showStatus("Cellpose: removing model…");
            IJ.showProgress(0.70);

            // run the python script. .init pre-imports cellpose (which pulls numpy) at
            // worker startup; importing it in-task hangs the worker on Windows (Appose #23).
            try (Service python = env.python().init("from cellpose import io")) {
                Map<String, Object> inputs = new HashMap<>();
                inputs.put("model_name", modelName);
                Service.Task task = python.task(script, inputs);
                task.waitFor();
            }

            IJ.showStatus("Cellpose: model succesfully removed!");
            IJ.showProgress(1.0);
        } catch (Exception e) {
            IJ.showProgress(1.0);
            IJ.handleException(e);
        }
    }
}
