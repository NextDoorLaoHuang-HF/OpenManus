import asyncio
import os
from contextlib import AsyncExitStack
from typing import List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from app.logger import logger
from app.tool.base import BaseTool, ToolResult
from app.tool.tool_collection import ToolCollection


class MCPClientTool(BaseTool):
    """Represents a tool proxy that can be called on the MCP server from the client side."""

    session: Optional[ClientSession] = None

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool by making a remote call to the MCP server."""
        if not self.session:
            return ToolResult(error="Not connected to MCP server")

        try:
            # 确保kwargs是字典类型
            if isinstance(kwargs, str):
                import json

                try:
                    kwargs = json.loads(kwargs)
                except json.JSONDecodeError:
                    return ToolResult(error=f"Invalid JSON in arguments: {kwargs}")

            result = await self.session.call_tool(self.name, kwargs)
            content_str = ", ".join(
                item.text for item in result.content if isinstance(item, TextContent)
            )
            return ToolResult(output=content_str or "No output returned.")
        except Exception as e:
            return ToolResult(error=f"Error executing tool: {str(e)}")


class MCPClients(ToolCollection):
    """
    A collection of tools that connects to an MCP server and manages available tools through the Model Context Protocol.
    """

    session: Optional[ClientSession] = None
    exit_stack: AsyncExitStack = None
    description: str = "MCP client tools for server interaction"

    def __init__(self):
        super().__init__()  # Initialize with empty tools list
        self.name = "mcp"  # Keep name for backward compatibility
        self.exit_stack = AsyncExitStack()

    async def connect_sse(self, server_url: str) -> None:
        """Connect to an MCP server using SSE transport."""
        if not server_url:
            raise ValueError("Server URL is required.")
        if self.session:
            await self.disconnect()

        streams_context = sse_client(url=server_url)
        streams = await self.exit_stack.enter_async_context(streams_context)
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(*streams)
        )

        await self._initialize_and_list_tools()

    async def connect_stdio(self, command: str, args: List[str]) -> None:
        server_params = StdioServerParameters(
            command=command, args=args, env=os.environ.copy()
        )
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )

        await self._initialize_and_list_tools()

    async def _initialize_and_list_tools(self) -> None:
        """Initialize session and populate tool map."""
        if not self.session:
            raise RuntimeError("Session not initialized.")

        await self.session.initialize()
        response = await self.session.list_tools()

        # Clear existing tools
        self.tools = tuple()
        self.tool_map = {}

        # Create proper tool objects for each server tool
        for tool in response.tools:
            server_tool = MCPClientTool(
                name=tool.name,
                description=tool.description,
                parameters=tool.inputSchema,
                session=self.session,
            )
            self.tool_map[tool.name] = server_tool

        self.tools = tuple(self.tool_map.values())
        logger.info(
            f"Connected to server with tools: {[tool.name for tool in response.tools]}"
        )

    async def disconnect(self) -> None:
        """Disconnect from the MCP server and clean up resources."""
        if self.session and self.exit_stack:
            # 首先清理会话和工具引用，确保即使exit_stack.aclose()失败也能释放资源
            session_copy = self.session
            exit_stack_copy = self.exit_stack

            # 立即清空引用，防止其他地方继续使用
            self.session = None
            self.tools = tuple()
            self.tool_map = {}

            try:
                # 直接关闭exit_stack，不使用独立任务，避免cancel scope问题
                try:
                    await exit_stack_copy.aclose()
                    logger.info("Disconnected from MCP server")
                except asyncio.CancelledError:
                    logger.warning("MCP disconnect was cancelled during aclose")
                except RuntimeError as re:
                    # 捕获cancel scope相关的RuntimeError
                    logger.error(f"RuntimeError during MCP disconnect: {re}")
            except Exception as e:
                logger.error(f"Error during MCP disconnect: {e}")
            finally:
                # 确保资源被释放
                logger.info("MCP resources have been released")
