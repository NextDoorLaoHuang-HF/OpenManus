import asyncio
import json
import os
import time
from typing import Dict  # Import Dict for type hinting

from app.agent.base import BaseAgent  # Import BaseAgent
from app.agent.manus import Manus
from app.agent.mcp import MCPAgent
from app.flow.flow_factory import FlowFactory, FlowType
from app.logger import logger


async def run_flow():
    # 初始化基本agent，并明确类型注解
    agents: Dict[str, BaseAgent] = {
        "manus": Manus(),
    }

    # 从配置文件加载MCP服务器配置
    try:
        config_path = os.path.join(
            os.path.dirname(__file__), "config", "mcp_servers.json"
        )
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)

            # 为每个MCP服务器创建对应的agent，但不立即初始化
            for server_name, server_config in config.get("mcpServers", {}).items():
                try:
                    # 创建MCPAgent实例
                    mcp_agent = MCPAgent()

                    # 设置环境变量
                    if "env" in server_config:
                        env_vars = server_config["env"]
                    else:
                        env_vars = {}

                    # 预初始化agent，获取能力信息但不保持连接
                    try:
                        # 获取模型配置（如果有）
                        model_config = server_config.get("model_config")

                        await mcp_agent.pre_initialize(
                            connection_type="stdio",
                            command=server_config.get("command"),
                            args=server_config.get("args", []),
                            env=env_vars,
                            model_config=model_config,
                        )
                        logger.info(
                            f"Pre-initialized MCP agent for {server_name} (capabilities loaded)"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to pre-initialize MCP agent for {server_name}: {str(e)}"
                        )
                        # 即使预初始化失败，仍然添加agent到列表中，后续可能会重试

                    # 将agent添加到agents字典中，使用服务器名称作为key
                    agents[server_name] = mcp_agent
                    logger.info(
                        f"Added MCP agent for {server_name} (will be fully initialized when needed)"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to create MCP agent for {server_name}: {str(e)}"
                    )
    except Exception as e:
        logger.error(f"Error loading MCP servers configuration: {str(e)}")

    try:
        prompt = input("Enter your prompt: ")

        if prompt.strip().isspace() or not prompt:
            logger.warning("Empty prompt provided.")
            return

        # 创建flow时，指定executors参数，包含所有可用的agent keys
        flow = FlowFactory.create_flow(
            flow_type=FlowType.PLANNING,
            agents=agents,
            executors=list(agents.keys()),  # 使用所有agent作为可能的执行器
        )
        logger.warning("Processing your request...")

        try:
            start_time = time.time()
            result = await asyncio.wait_for(
                flow.execute(prompt),
                timeout=3600,  # 60 minute timeout for the entire execution
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Request processed in {elapsed_time:.2f} seconds")
            logger.info(result)
        except asyncio.TimeoutError:
            logger.error("Request processing timed out after 1 hour")
            logger.info(
                "Operation terminated due to timeout. Please try a simpler request."
            )

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user.")
    except Exception as e:
        logger.error(f"Error: {str(e)}")

    # 清理MCP agents - 确保在所有步骤执行完毕后才进行清理
    logger.info("Starting cleanup of MCP agents...")
    for agent_name, agent in agents.items():
        if isinstance(agent, MCPAgent):
            try:
                logger.info(f"Attempting cleanup for MCP agent: {agent_name}")
                # 使用超时机制避免清理过程阻塞
                await asyncio.wait_for(agent.cleanup(), timeout=10.0)
                logger.info(f"Successfully cleaned up MCP agent: {agent_name}")
            except asyncio.TimeoutError:
                logger.warning(
                    f"Cleanup for MCP agent {agent_name} timed out after 10 seconds"
                )
                # 强制清理资源
                if hasattr(agent, "mcp_clients"):
                    agent.mcp_clients.session = None
                    agent.mcp_clients.tools = tuple()
                    agent.mcp_clients.tool_map = {}
                logger.info(f"Forced resource release for MCP agent: {agent_name}")
            except RuntimeError as re:
                # Specifically catch the RuntimeError related to cancel scopes
                logger.error(
                    f"RuntimeError during cleanup for MCP agent {agent_name}: {str(re)}"
                )
                # 强制清理资源
                if hasattr(agent, "mcp_clients"):
                    agent.mcp_clients.session = None
                    agent.mcp_clients.tools = tuple()
                    agent.mcp_clients.tool_map = {}
                logger.info(f"Forced resource release for MCP agent: {agent_name}")
            except Exception as e:
                # Catch other potential exceptions during cleanup
                logger.error(
                    f"General error cleaning up MCP agent {agent_name}: {type(e).__name__} - {str(e)}"
                )
                # 强制清理资源
                if hasattr(agent, "mcp_clients"):
                    agent.mcp_clients.session = None
                    agent.mcp_clients.tools = tuple()
                    agent.mcp_clients.tool_map = {}
                logger.info(f"Forced resource release for MCP agent: {agent_name}")
    logger.info("Finished cleanup of MCP agents.")


if __name__ == "__main__":
    asyncio.run(run_flow())
