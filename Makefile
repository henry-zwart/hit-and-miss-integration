MANDELBROT_FIGURE_NAMES = mandelbrot.png
MANDELBROT_DATA_NAMES = metadata.json hits.npy
MANDELBROT_FIGURES = $(patsubst %, figures/mandelbrot/%, $(MANDELBROT_FIGURE_NAMES))
MANDELBROT_DATA = $(patsubst %, data/mandelbrot/%, $(MANDELBROT_DATA_NAMES))

SHAPE_CONVERGENCE_FIGURE_NAMES = relchange_and_area.png
SHAPE_CONVERGENCE_DATA_NAMES = metadata.json iterations.npy area_cis.npy relchange_cis.npy
SHAPE_CONVERGENCE_FIGURES =  $(patsubst %, figures/shape_convergence/%, $(SHAPE_CONVERGENCE_FIGURE_NAMES))
SHAPE_CONVERGENCE_DATA =  $(patsubst %, data/shape_convergence/%, $(SHAPE_CONVERGENCE_DATA_NAMES))

LIMIT_CONVERGENCE_FIGURE_NAMES = limit_error.png
JOINT_CONVERGENCE_DATA_NAMES = metadata.json expected_err.npy err_confidence.npy
LIMIT_CONVERGENCE_FIGURES = $(patsubst %, figures/limit_convergence/%, $(LIMIT_CONVERGENCE_FIGURE_NAMES))
JOINT_CONVERGENCE_DATA = $(patsubst %, data/joint_convergence/%, $(JOINT_CONVERGENCE_DATA_NAMES))

SAMPLERS = random lhs ortho
SAMPLE_CONVERGENCE_FIGURE_NAMES = area.png convergent_samplesize_dist.png sampler_examples.png
SAMPLE_CONVERGENCE_FIGURES = $(patsubst %, figures/sample_convergence/%, $(SAMPLE_CONVERGENCE_FIGURE_NAMES))
SAMPLE_CONVERGENCE_DATA_NAMES = metadata.json $(patsubst %, %_measured_area.npy, $(SAMPLERS)) $(patsubst %, %_expected_area.npy, $(SAMPLERS)) $(patsubst %, %_ci.npy, $(SAMPLERS)) $(patsubst %, %_sample_size.npy, $(SAMPLERS))
SAMPLE_CONVERGENCE_DATA = $(patsubst %, data/sample_convergence/%, $(SAMPLE_CONVERGENCE_DATA_NAMES))

SAMPLE_ADAPTIVE_DATA_NAMES = metadata.json area.npy confidence_intervals.npy expected_area.npy total_samples.npy
SAMPLE_ADAPTIVE_DATA = $(patsubst %, data/sample_adaptive/%, $(SAMPLE_ADAPTIVE_DATA_NAMES))
FIGURES = $(MANDELBROT_FIGURES) $(SHAPE_CONVERGENCE_FIGURES) $(LIMIT_CONVERGENCE_FIGURES) $(JOINT_ERROR_FIGURES) $(SAMPLE_CONVERGENCE_FIGURES)

# all: $(SHAPE_CONVERGENCE_DATA)
all: $(FIGURES)

figures/%: | $(FIGURES_DIR)

$(SHAPE_CONVERGENCE_FIGURES) &: scripts/plot_shape_convergence.py $(SHAPE_CONVERGENCE_DATA)
	uv run $<

$(LIMIT_CONVERGENCE_FIGURES) &: scripts/plot_limit_convergence.py $(JOINT_CONVERGENCE_DATA)
	uv run $<

$(MANDELBROT_FIGURES) &: scripts/plot_mandelbrot.py $(MANDELBROT_DATA)
	uv run $<

$(SAMPLE_CONVERGENCE_FIGURES) &: scripts/plot_sample_convergence.py $(SAMPLE_CONVERGENCE_DATA) $(SAMPLE_ADAPTIVE_DATA)
	uv run $<

$(FIGURES_DIR):
	mkdir -p $@

$(SHAPE_CONVERGENCE_DATA) &: scripts/measure_shape_convergence.py | $(DATA_DIR)
	uv run $<

$(JOINT_CONVERGENCE_DATA) &: scripts/measure_joint_convergence.py $(SHAPE_CONVERGENCE_DATA) | $(DATA_DIR)
	uv run $<

$(MANDELBROT_DATA) &: scripts/deterministic_mandelbrot.py | $(DATA_DIR)
	uv run $<

$(SAMPLE_CONVERGENCE_DATA) &: scripts/measure_sample_convergence.py $(SHAPE_CONVERGENCE_DATA)
	uv run $<

$(DATA_DIR):
	mkdir -p $@

$(SAMPLE_ADAPTIVE_DATA) &: scripts/sample_adaptive.py $(SHAPE_CONVERGENCE_DATA)
	uv run $<


.PHONY: clean
clean:
	rm -rf data figures

