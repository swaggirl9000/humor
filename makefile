
qlogin_debug:
	qlogin -A CVLABPJ -q debug -b 1 -l elapstim_req=01:00:00 -T openmpi -v NQSV_MPI_VER=4.1.6/gcc11.4.0-cuda12.3.2 -V

download:
	HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download google/gemma-2-2b-it --exclude=*.gguf

run_notebook:
	@if [ -z "$(N)" ]; then \
		echo "Error: N variable is not set. Usage: make run_notebook N=path/to/notebook.ipynb"; \
		exit 1; \
	fi
	@N_PATH=$$(dirname "$(N)"); \
	N_NAME=$$(basename "$(N)"); \
	OUTPUT_DIR="$$N_PATH/outputs"; \
	mkdir -p "$$OUTPUT_DIR"; \
	echo "Running notebook: $(N)"; \
	HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 uv run papermill \
		$(N) \
		"$$OUTPUT_DIR/$$N_NAME" \
		--log-output \
		--progress-bar