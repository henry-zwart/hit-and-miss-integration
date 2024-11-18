FIGURES_DIR = results/figures
DATA_DIR = data

FIGURE_NAMES = \
	mandelbrot.png \
	relative_change.png \
	convergence_error.png \
	sampler_examples.png \
	sampler_estimates.png \
	sampler_convergence.png

FIGURES = $(patsubst %, results/figures/%, $(FIGURE_NAMES))

ENTRYPOINT ?= uv run

all: results/plot_metadata.json results/experiment_metadata.json

results/plot_metadata.json: scripts/plot_results.py results/experiment_metadata.json | $(FIGURES_DIR)
	$(ENTRYPOINT) $<

$(FIGURES_DIR):
	mkdir -p $@

results/experiment_metadata.json: \
			scripts/combine_metadata.py \
			data/mandelbrot/metadata.json \
			data/shape_convergence/metadata.json \
			data/joint_convergence/metadata.json \
			data/sample_convergence/metadata.json \
			data/sample_adaptive/metadata.json
	$(ENTRYPOINT) $<

data/mandelbrot/metadata.json: scripts/deterministic_mandelbrot.py | $(DATA_DIR)
	$(ENTRYPOINT) $<

data/shape_convergence/metadata.json: scripts/measure_shape_convergence.py | $(DATA_DIR)
	$(ENTRYPOINT) $<

data/joint_convergence/metadata.json: scripts/measure_joint_convergence.py data/shape_convergence/metadata.json | $(DATA_DIR)
	$(ENTRYPOINT) $<

data/sample_convergence/metadata.json: scripts/measure_sample_convergence.py data/shape_convergence/metadata.json | $(DATA_DIR)
	$(ENTRYPOINT) $<

data/sample_adaptive/metadata.json: scripts/sample_adaptive.py data/shape_convergence/metadata.json | $(DATA_DIR)
	$(ENTRYPOINT) $<

$(DATA_DIR):
	mkdir -p $@

.PHONY: clean
clean:
	rm -rf results data
