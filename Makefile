.ONESHELL:
SHELL = /bin/zsh

build: setup-pre-push
	@uv sync
	@echo "Finished building environment"

setup-pre-push:
	@cp pre_push.sh .git/hooks/pre-push
	@chmod +x .git/hooks/pre-push
	@echo "Done!"

run:
	@echo "Run with: uv run python main.py --config=<config>"

lint:
	@uv run black "src/"
	@uv run isort --settings-file=linters/isort.ini "src/"
	@uv run flake8 --config=linters/flake8.ini "src/"
	@echo "all done"
