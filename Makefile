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
	@PYTHONPATH=src uv run --with pytest --with pytest-asyncio pytest -q

test-debug:
	@PYTHONPATH=src uv run --with pytest --with pytest-asyncio pytest -vv

check:
	uv run --with mypy mypy src

migrate:
	uv run alembic upgrade head

api:
	uv run uvicorn admissions_conversation_engine.entrypoints.api:app --reload

token:
	python3 -c 'import os; from jose import jwt; print(jwt.encode({"sub":"agent-x"}, os.environ["AUTH__JWT_SECRET_KEY"], algorithm="HS256"))'