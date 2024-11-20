
package sc.fiji.labkit.cli;

import io.scif.codec.CompressionType;
import io.scif.config.SCIFIOConfig;
import io.scif.img.ImgSaver;
import io.scif.services.DatasetIOService;
import net.imglib2.view.Views;
import org.scijava.io.location.FileLocation;
import sc.fiji.labkit.cli.dilation.FastDilation;
import net.imagej.Dataset;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.labeling.ConnectedComponents;
import net.imglib2.algorithm.neighborhood.DiamondShape;
import net.imglib2.algorithm.neighborhood.RectangleShape;
import net.imglib2.img.ImgView;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.util.Cast;
import net.imglib2.util.Intervals;
import org.scijava.Context;
import org.scijava.command.CommandService;
import org.scijava.io.location.BytesLocation;
import org.scijava.util.ByteArray;
import picocli.CommandLine;
import sc.fiji.labkit.pixel_classification.RevampUtils;
import sc.fiji.labkit.ui.plugin.CalculateProbabilityMapWithLabkitPlugin;
import net.imglib2.img.Img;


import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Optional;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;

@CommandLine.Command(name = "probability",
		description = "Segment an image using a given classifier.")
public class ProbCommand implements Callable<Optional<Integer>> {

	@CommandLine.Option(names = { "--image" }, required = true,
			description = "Image to be segmented.")
	private File imageFile;

	@CommandLine.Option(names = { "--classifier" }, required = true,
			description = "Classifier that was trained using the FIJI Labkit plugin.")
	private File classifier;

	@CommandLine.Option(names = { "--output" }, required = true,
			description = "File to store the output image.")
	private File outputFile;

	@CommandLine.Option(names = { "--use-gpu" })
	private boolean useGpu = false;

	@Override
	public Optional<Integer> call() throws Exception {
		Context context = new Context();
		// services
		DatasetIOService datasetIoService = context.service(DatasetIOService.class);
		CommandService commandService = context.service(CommandService.class);
		System.out.println("open image");
		Dataset input = datasetIoService.open(imageFile.getAbsolutePath());
		System.out.println("Generate probability map");
		RandomAccessibleInterval<? extends RealType<?>> probabilityMap = Cast.unchecked(commandService.run(
				CalculateProbabilityMapWithLabkitPlugin.class, true, "input", input,
				"segmenter_file", classifier.getAbsolutePath(), "use_gpu", useGpu)
				.get().getOutput("output")); // Use correct key for probability map

		input = null;

		System.out.println("Write output probability map");
		writeImage(context, Cast.unchecked(probabilityMap));
		System.out.println("done");
		return Optional.of(0);
	}
	

	private <T extends RealType<T>> void writeImage(Context context, RandomAccessibleInterval<T> probabilityMap) throws IOException {
		try (FileOutputStream os = new FileOutputStream(outputFile)) {
			// Create an in-memory byte array to store the image
			ByteArray bytes = new ByteArray();
			
			// Configure SCIFIO image writer settings
			SCIFIOConfig config = new SCIFIOConfig();
			config.writerSetCompression("LZW"); // Use LZW compression for TIFF
			config.writerSetSequential(true);  // Enable sequential writing
			config.writerSetFailIfOverwriting(false); // Allow overwriting existing files
			
			// Wrap the probability map as an Img, required by ImgSaver
			Img<T> img = ImgView.wrap(probabilityMap, null);
			
			// Save the image to the byte array
			new ImgSaver(context).saveImg(new BytesLocation(bytes, outputFile.getName()), img, config);
			
			// Write the byte array to the actual file
			os.write(bytes.getArray(), 0, bytes.size());
		}
	}

}
