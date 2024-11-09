DATA = results/data
FIGURES = results/figures

TRUE_AREA_CONVERGENCE_PLOTS = \
			$(FIGURES)/true_area_conv.png \
			$(FIGURES)/true_area_conv_closeup.png

DATA_PREREQS = .targets/true_area_convergence

all: $(TRUE_AREA_CONVERGENCE_PLOTS)


$(FIGURES)/%: scripts/plot_results.py $(DATA_PREREQS)
	mkdir -p $(FIGURES) && \
	uv run scripts/plot_results.py

.targets/true_area_convergence: scripts/mb_iter_convergence.py $(DATA) .targets
	mkdir -p results/data/true_area_convergence && \
	uv run scripts/mb_iter_convergence.py && \
	touch $@

$(DATA):
	mkdir -p $@

.targets:
	mkdir $@

.PHONY: clean
clean:
	rm -rf .targets results



