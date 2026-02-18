# Define the path to your .env file
ENV_FILE := .env

# Include .env file if it exists
ifneq (,$(wildcard $(ENV_FILE)))
    include $(ENV_FILE)
    export $(shell sed 's/=.*//' $(ENV_FILE))
endif

dev:
	@uv run langgraph dev

test:
	@PYTHONPATH=src uv run --with pytest pytest -q

test-debug:
	@PYTHONPATH=src uv run --with pytest pytest -vv

check:
	uv run --with mypy mypy src