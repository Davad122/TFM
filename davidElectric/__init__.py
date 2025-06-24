from .server import mcp

def main() -> None:
    "Run the MCP server"
    mcp.run(transport='stdio')