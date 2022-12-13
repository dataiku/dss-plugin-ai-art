# Makefile variables set automatically
plugin_id=ai-art
plugin_version=`cat plugin.json | python -c "import sys, json; print(str(json.load(sys.stdin)['version']).replace('/',''))"`
archive_file_name="dss-plugin-${plugin_id}-${CUDA_VERSION_ID}-${plugin_version}.zip"
remote_url=`git config --get remote.origin.url`
last_commit_id=`git rev-parse "$${git_stash:-HEAD}"`

# You can override the CUDA version by setting the $CUDA_VERSION env-var
CUDA_VERSION ?= 10.2
# Remove '.' from $CUDA_VERSION, e.g. '10.2' > 'cu102'
CUDA_VERSION_ID = cu$(subst .,,$(CUDA_VERSION))


plugin:
	@echo "[START] Archiving plugin to dist/ folder..."
	@echo "CUDA version: ${CUDA_VERSION} (${CUDA_VERSION_ID})"
	@sed -i "s/${plugin_id}-cu[[:digit:]]\+/${plugin_id}-${CUDA_VERSION_ID}/" plugin.json
	@sed -i "s/CUDA [\.[:digit:]]\+/CUDA ${CUDA_VERSION}/" plugin.json
	@cat plugin.json | json_pp > /dev/null
	@sed -i "s/cu[[:digit:]]\+/${CUDA_VERSION_ID}/" code-env/python/spec/requirements.txt
	@rm -rf dist
	@mkdir dist
#	Stash the unstaged changes made above so that git-archive picks them up
	@( \
		git_stash="$$(git stash create)"; \
		echo "Stashed changes to $${git_stash:-HEAD}"; \
		echo "{\"remote_url\":\"${remote_url}\",\"last_commit_id\":\"${last_commit_id}\"}" > release_info.json; \
		git archive -v -9 --format zip -o dist/${archive_file_name} -- "$${git_stash:-HEAD}"; \
	)
	@zip --delete dist/${archive_file_name} "tests/*" "doc/*"
	@zip -u dist/${archive_file_name} release_info.json
	@rm release_info.json
	@echo "[SUCCESS] Archiving plugin to dist/ folder: Done!"

unit-tests:
	@echo "Running unit tests..."
	@( \
		PYTHON_VERSION=`python3 -c "import sys; print('PYTHON{}{}'.format(sys.version_info.major, sys.version_info.minor))"`; \
		PYTHON_VERSION_IS_CORRECT=`cat code-env/python/desc.json | python3 -c "import sys, json; print('$$PYTHON_VERSION' in json.load(sys.stdin)['acceptedPythonInterpreters']);"`; \
		if [ $$PYTHON_VERSION_IS_CORRECT == "False" ]; then echo "Python version $$PYTHON_VERSION is not in acceptedPythonInterpreters"; exit 1; else echo "Python version $$PYTHON_VERSION is in acceptedPythonInterpreters"; fi; \
	)
	@( \
		rm -rf ./env/; \
		python3 -m venv env/; \
		source env/bin/activate; \
		pip install --upgrade pip;\
		pip install --no-cache-dir -r tests/python/unit/requirements.txt; \
		pip install --no-cache-dir -r code-env/python/spec/requirements.txt; \
		export PYTHONPATH="$(PYTHONPATH):$(PWD)/python-lib"; \
		pytest tests/python/unit --alluredir=tests/allure_report || ret=$$?; exit $$ret \
	)

integration-tests:
	@echo "Running integration tests..."
	@( \
		rm -rf ./env/; \
		python3 -m venv env/; \
		source env/bin/activate; \
		pip3 install --upgrade pip;\
		pip install --no-cache-dir -r tests/python/integration/requirements.txt; \
		pytest tests/python/integration --alluredir=tests/allure_report || ret=$$?; exit $$ret \
	)

lint:
	@echo "Running linters..."
	@python3 -m venv --clear env
	@( \
		. env/bin/activate && \
		pip3 install --upgrade pip && \
		pip3 install -r requirements-lint.txt && \
		echo "Running Flake8..." && \
		flake8 . && \
		echo "Running Black..." && \
		black --check . \
	)

black:
	@echo "Formatting code using Black..."
#	Stash any unstaged changes so that we can revert back if needed
	@( \
		git_stash="$$(git stash create)" && \
		echo "Stashed changes to $${git_stash:-HEAD}" \
	)
	@python3 -m venv --clear env
	@( \
		. env/bin/activate && \
		pip3 install --upgrade pip && \
		pip3 install -r requirements-lint.txt && \
		echo "Running Black..." && \
		black . \
	)

tests: lint unit-tests integration-tests

dist-clean:
	rm -rf dist
