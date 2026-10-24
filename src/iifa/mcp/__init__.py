"""Door 1: the MCP server (SPEC section 9). See ``iifa.mcp.server`` for the tools,
``iifa.mcp.http`` for the endpoint mounted on the service and ``python -m iifa.mcp``
for the stdio process."""

from iifa.mcp.server import CORE_TOOLS, FORGET_TOOL, MCPConfig, build_server

__all__ = ["CORE_TOOLS", "FORGET_TOOL", "MCPConfig", "build_server"]
