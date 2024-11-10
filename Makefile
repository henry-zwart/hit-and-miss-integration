SHAPE_CONVERGENCE_FIGURE_NAMES = relchange_and_area.png area_after_convergence.png
SHAPE_CONVERGENCE_DATA_NAMES = metadata.json iterations.npy area_cis.npy relchange_cis.npy
SHAPE_CONVERGENCE_FIGURES =  $(patsubst %, figures/shape_convergence/%, $(SHAPE_CONVERGENCE_FIGURE_NAMES))
SHAPE_CONVERGENCE_DATA =  $(patsubst %, data/shape_convergence/%, $(SHAPE_CONVERGENCE_DATA_NAMES))

FIGURES = $(SHAPE_CONVERGENCE_FIGURES)

# all: $(SHAPE_CONVERGENCE_DATA)
all: $(FIGURES)

$(SHAPE_CONVERGENCE_FIGURES) &: scripts/plot_shape_convergence.py $(SHAPE_CONVERGENCE_DATA) | $(FIGURES_DIR)
	uv run $<

$(FIGURES_DIR):
	mkdir -p $@

$(SHAPE_CONVERGENCE_DATA) &: scripts/measure_shape_convergence.py | $(DATA_DIR)
	uv run $<

$(DATA_DIR):
	mkdir -p $@

.PHONY: clean
clean:
	rm -rf data figures

