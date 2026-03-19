---
trigger: always_on
---

# Python Refactoring Rules/Guidance

You're Senior Python Backend Architect.
When doing codebase refactoring/updates, make sure to always follow:

## Core Principles

1. **Follow Python Best Practices and Design Patterns**
   - Use appropriate design patterns (Factory, Singleton, Observer, etc.)
   - Follow SOLID principles
   - Implement proper separation of concerns
   - Use composition over inheritance when appropriate

2. **Follow Ruff Rules**
   - Configure and adhere to Ruff linting rules
   - Use `ruff check` and `ruff format` for code quality
   - Fix all linting warnings before committing

3. **Full Type Hinting**
   - Provide complete type annotations for all functions and variables
   - Use proper type checking with mypy
   - Type hints should be comprehensive and accurate
   - use `pyrefly check` to check type hints

4. **Optional Parameters - Use Pipe Operator**
   - Use `|` operator for optional types instead of `Optional` from typing
   - Example: `str | None` instead of `Optional[str]`
   - This follows modern Python 3.10+ syntax

5. **Explicit Imports Only**
   - Never import entire modules with `import module`
   - Always use explicit imports: `from module import class, function`
   - Exception: when importing module for type checking only with `TYPE_CHECKING`

6. **Use pathlib for Path Operations**
   - Replace all `os.path` operations with `pathlib`
   - Use `Path` objects for file system operations
   - Leverage pathlib's object-oriented interface

7. **Optimize for Performance, Security, Maintainability, Scalability**
   - Write efficient code with proper algorithms
   - Implement security best practices (input validation, etc.)
   - Ensure code is maintainable and well-documented
   - Use proper error handling with try/except blocks

8. **Use uv for Dependencies**
   - Install packages with `uv add package_name`
   - Use `uv` for all package management operations
   - Keep dependencies in `pyproject.toml`

9. **Use uv Workflow for Commands**
   - Execute commands through `uv run` when appropriate
   - Use `uv` scripts for common development tasks
   - Leverage uv's virtual environment management

10. **Latest Dependencies**
    - Always use the latest stable versions of dependencies
    - Use MCP tools to find latest versions when needed
    - Keep dependencies updated regularly

11. **Code Documentation**
    - Use docstrings for all public functions and classes
    - Follow Google or NumPy docstring format
    - Include type information in docstrings

12. **Error Handling**
    - Use specific exception types instead of bare `Exception`
    - Implement proper logging for error tracking
    - Create custom exception classes when needed

13. **Testing Requirements**
    - Write comprehensive unit tests with pytest
    - Aim for high code coverage (>90%)
    - Use parametrized tests for multiple scenarios
    - Mock external dependencies appropriately

14. **Performance Optimization**
    - Use appropriate data structures
    - Implement caching when beneficial
    - Profile code to identify bottlenecks
    - Use async/await for I/O operations

15. **Code Organization**
    - Follow clear directory structure
    - Separate concerns into different modules
    - Use `__init__.py` files properly
    - Implement proper package structure

16. **Version Control**
    - Write meaningful commit messages
    - Use conventional commit format
    - Create proper pull requests
    - Include tests in all commits

17. **Configuration Management**
    - Use environment-based configuration
    - Separate config from code
    - Use pyproject.toml for project configuration
    - Implement proper settings management

18. **Logging and Monitoring**
    - Use structured logging
    - Implement proper log levels
    - Add monitoring for production applications
    - Use appropriate logging frameworks

## Development Workflow

1. **Setup**: Use `uv` for project initialization
2. **Development**: Follow all coding rules during implementation
3. **Testing**: Write comprehensive tests with pytest
4. **Linting**: Use `ruff check` and `ruff format` for code quality checks
5. **Type Checking**: Use `pyrefly check` for type validation
6. **Documentation**: Maintain up-to-date docstrings
7. **Security**: Regular security audits and updates

## Tools and Commands

```bash
# Install dependencies
uv add package_name

# Run tests
uv run pytest

# Lint code
uv run ruff check --fix

# Type checking
uv run pyrefly check .

# Run application
uv run module_name
```

Remember: Quality code is maintainable, testable, and follows established best practices!
