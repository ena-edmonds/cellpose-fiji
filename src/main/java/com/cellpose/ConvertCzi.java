package com.cellpose;

import ij.IJ;
import ij.io.DirectoryChooser;
import org.scijava.command.Command;
import org.scijava.plugin.Plugin;
import java.io.File;

import ij.ImagePlus;
import ij.plugin.ZProjector;
import loci.plugins.BF;
import loci.plugins.in.ImporterOptions;
import loci.common.DebugTools;

@Plugin(type = Command.class, menuPath = "Process>Batch Flatten Stacks")
public class ConvertCzi implements Command {
    @Override
    public void run() {
        DebugTools.setRootLevel("ERROR"); // make the console not get spammed with a bunch of warnings

        DirectoryChooser dc = new DirectoryChooser("Choose a folder of czi files to convert");
        File directory = new File(dc.getDirectory());
        if (dc.getDirectory() == null) return; // canceled
        for(File czi : directory.listFiles()) {
            // if the file is a CZI and the equivalent projection .png doesn't exist, process it
            if(isCzi(czi) && !(new File(directory, czi.getName().replaceAll(".czi", "_MAX.png")).exists())) try {
                processCzi(czi); 
            } catch (Exception e ) {
                IJ.log("Failed: " + czi.getName() + " (" + e.getMessage() + ")");
            }
        }
    }

    private void processCzi(File czi) throws Exception {
        ImporterOptions options = new ImporterOptions();
        options.setId(czi.getAbsolutePath());
        options.setColorMode(ImporterOptions.COLOR_MODE_COMPOSITE);
        options.setQuiet(true);
        ImagePlus imp = BF.openImagePlus(options)[0];

        // Equivalent to run("Z Project...", "projection=[Max Intensity]")
        ImagePlus projected = imp.getNSlices() > 1 ? ZProjector.run(imp, "max") : imp;

        File out = new File(czi.getParentFile(), czi.getName().substring(0, czi.getName().indexOf(".")) + "_MAX" + ".png");
        IJ.saveAs(projected, "PNG", out.getAbsolutePath());

        if (projected != imp) projected.close();
        imp.close();
    }

    private boolean isCzi(File f) { return f.getName().toLowerCase().endsWith(".czi"); }
}
