MANDELBROT_FIGURE_NAMES = mandelbrot.png
MANDELBROT_DATA_NAMES = metadata.json hits.npy
MANDELBROT_FIGURES = $(patsubst %, figures/mandelbrot/%, $(MANDELBROT_FIGURE_NAMES))
MANDELBROT_DATA = $(patsubst %, data/mandelbrot/%, $(MANDELBROT_DATA_NAMES))

SHAPE_CONVERGENCE_FIGURE_NAMES = relchange_and_area.png area_after_convergence.png
SHAPE_CONVERGENCE_DATA_NAMES = metadata.json iterations.npy area_cis.npy relchange_cis.npy
SHAPE_CONVERGENCE_FIGURES =  $(patsubst %, figures/shape_convergence/%, $(SHAPE_CONVERGENCE_FIGURE_NAMES))
SHAPE_CONVERGENCE_DATA =  $(patsubst %, data/shape_convergence/%, $(SHAPE_CONVERGENCE_DATA_NAMES))

LIMIT_CONVERGENCE_FIGURE_NAMES = limit_error.png limit_area.png
JOINT_CONVERGENCE_DATA_NAMES = metadata.json expected_area.npy confidence_intervals.npy
LIMIT_CONVERGENCE_FIGURES = $(patsubst %, figures/limit_convergence/%, $(LIMIT_CONVERGENCE_FIGURE_NAMES))
JOINT_CONVERGENCE_DATA = $(patsubst %, data/joint_convergence/%, $(JOINT_CONVERGENCE_DATA_NAMES))

JOINT_ERROR_FIGURE_NAMES = equiv_error_areas.png error_contour.png error_hist.png
JOINT_ERROR_FIGURES = $(patsubst %, figures/joint_error/%, $(JOINT_ERROR_FIGURE_NAMES))

FIGURES = $(MANDELBROT_FIGURES) $(SHAPE_CONVERGENCE_FIGURES) $(LIMIT_CONVERGENCE_FIGURES) $(JOINT_ERROR_FIGURES)

# all: $(SHAPE_CONVERGENCE_DATA)
all: $(FIGURES)

figures/%: | $(FIGURES_DIR)

$(SHAPE_CONVERGENCE_FIGURES) &: scripts/plot_shape_convergence.py $(SHAPE_CONVERGENCE_DATA)
	uv run $<

$(LIMIT_CONVERGENCE_FIGURES) &: scripts/plot_limit_convergence.py $(JOINT_CONVERGENCE_DATA)
	uv run $<

$(JOINT_ERROR_FIGURES) &: scripts/plot_joint_error.py $(JOINT_CONVERGENCE_DATA)
	uv run $<

$(MANDELBROT_FIGURES) &: scripts/plot_mandelbrot.py $(MANDELBROT_DATA)
	uv run $<

$(FIGURES_DIR):
	mkdir -p $@

$(SHAPE_CONVERGENCE_DATA) &: scripts/measure_shape_convergence.py | $(DATA_DIR)
	uv run $<

$(JOINT_CONVERGENCE_DATA) &: scripts/measure_joint_convergence.py src/hit_and_mandelbrot/mandelbrot.py | $(DATA_DIR)
	uv run $<

$(MANDELBROT_DATA) &: scripts/deterministic_mandelbrot.py | $(DATA_DIR)
	uv run $<

$(DATA_DIR):
	mkdir -p $@

.PHONY: clean
clean:
	rm -rf data figures

