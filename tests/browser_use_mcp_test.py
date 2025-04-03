import asyncio
import logging
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent.mcp import MCPAgent

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_browser_use_service():
    """测试浏览器使用MCP服务"""
    try:
        # 设置环境变量
        import json

        with open(
            "/Users/huangweihao/openmanus/OpenManus/config/mcp_servers.json", "r"
        ) as f:
            config = json.load(f)

        # 设置所有环境变量
        for key, value in config["mcpServers"]["browser-use"]["env"].items():
            os.environ[key] = value

        agent = MCPAgent()

        # 提取browser-use服务的连接配置
        server_config = config["mcpServers"]["browser-use"]
        command = server_config.get("command")
        args = server_config.get("args")

        # 初始化agent
        await agent.initialize(
            connection_type="stdio",
            command=command,
            args=args,
        )

        # 测试浏览器相关功能
        test_queries = [
            "打开百度搜索最新新闻，并总结输出",
        ]

        for query in test_queries:
            logger.info(f"正在执行: {query}")
            result = await agent.run(query)
            logger.info(f"执行结果: {result}")

    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        raise
    finally:
        if "agent" in locals():
            await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(test_browser_use_service())
