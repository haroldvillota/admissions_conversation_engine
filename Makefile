# Define the path to your .env file
ENV_FILE := .env

# Include .env file if it exists
ifneq (,$(wildcard $(ENV_FILE)))
    include $(ENV_FILE)
    export $(shell sed 's/=.*//' $(ENV_FILE))
endif

dev:
	@uv run langgraph dev

