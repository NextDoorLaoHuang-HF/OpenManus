import asyncio
import os
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.logger import logger
from app.prompt.mcp import MULTIMEDIA_RESPONSE_PROMPT, NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import AgentState, Message
from app.tool.base import ToolResult
from app.tool.mcp import MCPClients


class MCPAgent(ToolCallAgent):
    """Agent for interacting with MCP (Model Context Protocol) servers.

    This agent connects to an MCP server using either SSE or stdio transport
    and makes the server's tools available through the agent's tool interface.
    """

    name: str = "mcp_agent"
    description: str = "An agent that connects to an MCP server and uses its tools."

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    # Initialize MCP tool collection
    mcp_clients: MCPClients = Field(default_factory=MCPClients)
    available_tools: MCPClients = None  # Will be set in initialize()

    # 存储服务器配置信息，用于延迟初始化
    server_config: dict = Field(default_factory=dict)

    max_steps: int = 10
    connection_type: str = "stdio"  # "stdio" or "sse"

    # Track tool schemas to detect changes
    tool_schemas: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    _refresh_tools_interval: int = 5  # Refresh tools every N steps

    # Special tool names that should trigger termination
    special_tool_names: List[str] = Field(default_factory=lambda: ["terminate"])

    async def pre_initialize(
        self,
        connection_type: Optional[str] = None,
        server_url: Optional[str] = None,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[dict] = None,
        model_config: Optional[dict] = None,
    ) -> None:
        """预初始化MCP连接，获取能力信息后断开连接。

        此方法用于规划阶段，只获取MCP服务器的能力信息，不保持长连接。

        Args:
            connection_type: 连接类型 ("stdio" 或 "sse")
            server_url: MCP服务器URL (用于SSE连接)
            command: 要运行的命令 (用于stdio连接)
            args: 命令的参数 (用于stdio连接)
            env: 环境变量 (用于stdio连接)
            model_config: 模型配置 (用于指定使用的LLM模型)
        """
        # 存储配置信息，用于后续完整初始化
        self.server_config = {
            "connection_type": connection_type or self.connection_type,
            "server_url": server_url,
            "command": command,
            "args": args or [],
            "env": env or {},
            "model_config": model_config or {},
        }

        # 设置环境变量
        if env:
            for key, value in env.items():
                os.environ[key] = value

        # 临时连接到MCP服务器获取能力信息
        try:
            logger.info(f"预初始化MCP代理: {self.name}")

            # 根据连接类型连接到MCP服务器
            if connection_type:
                self.connection_type = connection_type

            if self.connection_type == "sse":
                if not server_url:
                    raise ValueError("SSE连接需要提供服务器URL")
                await self.mcp_clients.connect_sse(server_url=server_url)
            elif self.connection_type == "stdio":
                if not command:
                    raise ValueError("stdio连接需要提供命令")
                await self.mcp_clients.connect_stdio(command=command, args=args or [])
            else:
                raise ValueError(f"不支持的连接类型: {self.connection_type}")

            # 设置可用工具
            self.available_tools = self.mcp_clients

            # 合成描述信息
            await self._synthesize_description()

            # 刷新工具列表
            await self._refresh_tools()

            logger.info(f"MCP代理预初始化完成: {self.name}")
        finally:
            # 无论成功与否，都断开连接
            await self.cleanup()
            logger.info(f"MCP代理预初始化后断开连接: {self.name}")

    async def initialize(
        self,
        connection_type: Optional[str] = None,
        server_url: Optional[str] = None,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        model_config: Optional[dict] = None,
    ) -> None:
        """完全初始化MCP连接。

        Args:
            connection_type: 连接类型 ("stdio" 或 "sse")
            server_url: MCP服务器URL (用于SSE连接)
            command: 要运行的命令 (用于stdio连接)
            args: 命令的参数 (用于stdio连接)
            model_config: 模型配置 (用于指定使用的LLM模型)
        """
        if connection_type:
            self.connection_type = connection_type

        # 如果提供了模型配置，使用指定的模型初始化LLM
        if model_config and model_config.get("model"):
            # 使用服务器名称作为配置名称，并创建新的LLM实例
            self.llm = LLM(
                config_name=self.name.lower(), llm_config={"default": model_config}
            )
            logger.info(
                f"为MCP代理 {self.name} 使用自定义模型: {model_config.get('model')}"
            )

        # 根据连接类型连接到MCP服务器
        if self.connection_type == "sse":
            if not server_url:
                raise ValueError("SSE连接需要提供服务器URL")
            await self.mcp_clients.connect_sse(server_url=server_url)
        elif self.connection_type == "stdio":
            if not command:
                raise ValueError("stdio连接需要提供命令")
            await self.mcp_clients.connect_stdio(command=command, args=args or [])
        else:
            raise ValueError(f"不支持的连接类型: {self.connection_type}")

        # 设置可用工具
        self.available_tools = self.mcp_clients

        # 合成描述信息
        await self._synthesize_description()

        # Store initial tool schemas
        await self._refresh_tools()  # _refresh_tools also updates self.tool_schemas

        # Add system message about available tools AND the agent's (potentially updated) description
        tool_names = list(self.mcp_clients.tool_map.keys())
        tools_info = ", ".join(tool_names) if tool_names else "No tools reported."

        # Construct the final system message including the potentially synthesized description
        system_message_content = f"{self.system_prompt}\n\nAgent Description: {self.description}\n\nAvailable MCP tools: {tools_info}"

        self.memory.add_message(Message.system_message(system_message_content))

    async def _refresh_tools(self) -> Tuple[List[str], List[str]]:
        """Refresh the list of available tools from the MCP server.

        Returns:
            A tuple of (added_tools, removed_tools)
        """
        if not self.mcp_clients.session:
            return [], []

        # Get current tool schemas directly from the server
        response = await self.mcp_clients.session.list_tools()
        current_tools = {tool.name: tool.inputSchema for tool in response.tools}

        # Determine added, removed, and changed tools
        current_names = set(current_tools.keys())
        previous_names = set(self.tool_schemas.keys())

        added_tools = list(current_names - previous_names)
        removed_tools = list(previous_names - current_names)

        # Check for schema changes in existing tools
        changed_tools = []
        for name in current_names.intersection(previous_names):
            if current_tools[name] != self.tool_schemas.get(name):
                changed_tools.append(name)

        # Update stored schemas
        self.tool_schemas = current_tools

        # Log and notify about changes
        if added_tools:
            logger.info(f"Added MCP tools: {added_tools}")
            self.memory.add_message(
                Message.system_message(f"New tools available: {', '.join(added_tools)}")
            )
        if removed_tools:
            logger.info(f"Removed MCP tools: {removed_tools}")
            self.memory.add_message(
                Message.system_message(
                    f"Tools no longer available: {', '.join(removed_tools)}"
                )
            )
        if changed_tools:
            logger.info(f"Changed MCP tools: {changed_tools}")

        return added_tools, removed_tools

    async def think(self) -> bool:
        """Process current state and decide next action."""
        # Check MCP session and tools availability
        if not self.mcp_clients.session or not self.mcp_clients.tool_map:
            logger.info("MCP service is no longer available, ending interaction")
            self.state = AgentState.FINISHED
            return False

        # Refresh tools periodically
        if self.current_step % self._refresh_tools_interval == 0:
            await self._refresh_tools()
            # All tools removed indicates shutdown
            if not self.mcp_clients.tool_map:
                logger.info("MCP service has shut down, ending interaction")
                self.state = AgentState.FINISHED
                return False

        # Use the parent class's think method to get LLM response and tool calls
        should_continue = await super().think()

        # Check for TASK_COMPLETE signal regardless of tool calls
        completion_signal = "TASK_COMPLETE"
        if self.memory.messages:
            last_message = self.memory.messages[-1]
            # --- Add detailed logging before the check ---
            logger.debug(
                f"Checking for completion signal. should_continue={should_continue}"
            )
            if last_message.role == "assistant" and last_message.content:
                content_stripped = last_message.content.strip()
                contains_signal = completion_signal in content_stripped
                logger.debug(f"Last message role: {last_message.role}")
                logger.debug(
                    f"Last message content (raw): {repr(last_message.content)}"
                )  # Use repr() to see hidden chars
                logger.debug(
                    f"Last message content (stripped): {repr(content_stripped)}"
                )  # Use repr()
                logger.debug(f"Completion signal: '{completion_signal}'")
                logger.debug(f"Signal found in stripped content? {contains_signal}")

                # --- Existing Check: Use 'in' on stripped content ---
                if contains_signal:
                    logger.info(
                        f"LLM indicated task completion via content signal: '{completion_signal}' found in content."
                    )
                    self.state = AgentState.FINISHED
                    logger.debug(
                        f"State set to: {self.state}. Returning False."
                    )  # Log state change
                    # Ensure we return False to stop the loop
                    return False
                else:
                    logger.debug("Completion signal check failed.")  # Log failure
            else:
                logger.debug(
                    "Last message not from assistant or no content."
                )  # Log reason for skipping check
        # --- End detailed logging ---

        # Return the original continuation status determined by super().think()
        logger.debug(
            f"Think method returning should_continue={should_continue}"
        )  # Log return value
        return should_continue

    async def _handle_special_tool(self, name: str, result: Any, **kwargs) -> None:
        """Handle special tool execution and state changes"""
        # First process with parent handler
        await super()._handle_special_tool(name, result, **kwargs)

        # Handle multimedia responses
        if isinstance(result, ToolResult) and result.base64_image:
            self.memory.add_message(
                Message.system_message(
                    MULTIMEDIA_RESPONSE_PROMPT.format(tool_name=name)
                )
            )

    def _should_finish_execution(self, name: str, **kwargs) -> bool:
        """Determine if tool execution should finish the agent"""
        # Terminate if the tool name is 'terminate'
        return name.lower() == "terminate"

    async def _synthesize_description(self) -> None:
        """从MCP服务器的工具中合成代理描述。"""
        if self.mcp_clients.tool_map:
            tool_descriptions = []
            for tool_name, tool_instance in self.mcp_clients.tool_map.items():
                # 使用MCP服务器提供的工具描述
                desc = getattr(tool_instance, "description", "No description provided")
                tool_descriptions.append(f"{tool_name}: {desc}")

            if tool_descriptions:
                synthesized_description = (
                    "An agent connected to an MCP server providing the following tools:\n- "
                    + "\n- ".join(tool_descriptions)
                )
                self.description = synthesized_description
                logger.info(
                    f"Synthesized description for {self.name} based on MCP tools."
                )
            else:
                logger.warning(
                    f"MCP server for {self.name} reported tools, but failed to generate descriptions. Using default."
                )
        else:
            logger.warning(
                f"No tools found on MCP server for {self.name}. Using default description: '{self.description}'"
            )

    async def cleanup(self) -> None:
        """Clean up MCP connection when done."""
        if self.mcp_clients.session:
            try:
                # 添加超时机制，避免无限等待
                await asyncio.wait_for(self.mcp_clients.disconnect(), timeout=5.0)
                logger.info("MCP connection closed")
            except asyncio.TimeoutError:
                logger.warning("MCP cleanup timed out after 5 seconds, forcing cleanup")
                # 确保资源被释放
                self.mcp_clients.session = None
                self.mcp_clients.tools = tuple()
                self.mcp_clients.tool_map = {}
            except asyncio.exceptions.CancelledError:
                logger.warning("MCP cleanup was cancelled, forcing resource release")
                # 确保资源被释放
                self.mcp_clients.session = None
                self.mcp_clients.tools = tuple()
                self.mcp_clients.tool_map = {}
            except Exception as e:
                logger.error(f"Error during MCP cleanup: {e}")
                # 确保资源被释放
                self.mcp_clients.session = None
                self.mcp_clients.tools = tuple()
                self.mcp_clients.tool_map = {}

    async def run(self, request: Optional[str] = None) -> str:
        """Run the agent without automatic cleanup to maintain connection for multi-step execution.

        The cleanup will be handled by the flow after all steps are completed.
        """
        # 不再在每次run后自动调用cleanup，避免连接提前关闭
        result = await super().run(request)
        return result
