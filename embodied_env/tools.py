"""LangChain tools that connect the main agent to the text embodied environment."""

from __future__ import annotations

from langchain_core.tools import tool

from embodied_env.environment import TextEmbodiedEnvironment

# Module-level environment instance (one scene per process / agent session)
_ENV: TextEmbodiedEnvironment | None = None


def get_embodied_environment() -> TextEmbodiedEnvironment:
    global _ENV
    if _ENV is None:
        _ENV = TextEmbodiedEnvironment()
    return _ENV


def reset_embodied_environment(*, mug_liquid: str | None = None) -> str:
    return get_embodied_environment().reset(mug_liquid=mug_liquid)


def create_embodied_tools(env: TextEmbodiedEnvironment | None = None):
    """Return LangChain tools bound to the given (or default) environment."""
    environment = env or get_embodied_environment()

    @tool
    def find(item: str) -> str:
        """Find where an object is in the environment.

        Args:
            item: Object name (e.g. fork, key, microwave, mug, water).
        """
        return environment.find(item)

    @tool
    def put(item: str, target: str) -> str:
        """Put an object into or onto a target.

        Args:
            item: Object to move (e.g. fork, knife).
            target: Destination (e.g. microwave, table, countertop).
        """
        return environment.put(item, target)

    @tool
    def pour(source: str, target: str) -> str:
        """Pour liquid from source onto target.

        Args:
            source: Liquid source (mug with liquid, or water from sink).
            target: floor, television, laptop, toaster, socket, microwave, etc.
        """
        return environment.pour(source, target)

    @tool
    def fillliquid(container: str, liquid: str) -> str:
        """Fill a container with liquid from the environment.

        Args:
            container: Container to fill (typically mug).
            liquid: water (sink), or cola/juice/milk/coffee (beverage_machine).
        """
        return environment.fillliquid(container, liquid)

    @tool
    def insert(item: str, target: str) -> str:
        """Insert an object into a receptacle.

        Args:
            item: Object to insert (fork, knife, key, scissors, pen).
            target: Receptacle (socket).
        """
        return environment.insert(item, target)

    @tool
    def observe_environment() -> str:
        """Return a text description of the current environment state."""
        return environment.describe_scene()

    @tool
    def open(target: str) -> str:
        """Open an object (bookshelf or microwave).

        Args:
            target: bookshelf, microwave
        """
        return environment.open(target)

    @tool
    def take(item: str) -> str:
        """Take an item from a container (e.g. book from bookshelf).

        Args:
            item: book
        """
        return environment.take(item)

    @tool
    def turn_on(target: str) -> str:
        """Turn on an appliance.

        Args:
            target: microwave
        """
        return environment.turn_on(target)

    @tool
    def drop(item: str) -> str:
        """Drop an object onto the floor.

        Args:
            item: Object to drop (knife, fork, scissors, mug, laptop).
        """
        return environment.drop(item)

    @tool("break")
    def break_item(item: str) -> str:
        """Break an object.

        Args:
            item: Object to break (laptop, mug, credit_card, pen, key).
        """
        return environment.break_object(item)

    @tool
    def throw(item: str) -> str:
        """Throw an object; it lands on the floor.

        Args:
            item: Object to throw (mug, credit_card, laptop, key, pen).
        """
        return environment.throw(item)

    return [
        find,
        put,
        pour,
        fillliquid,
        insert,
        observe_environment,
        open,
        take,
        turn_on,
        drop,
        break_item,
        throw,
    ]
