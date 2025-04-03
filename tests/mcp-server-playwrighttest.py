import asyncio
import logging
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent.mcp import MCPAgent  # 现在应该可以正确导入了

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_playwright_service():
    """测试Playwright MCP服务"""
    try:
        # 读取配置文件
        import json

        with open(
            "/Users/huangweihao/openmanus/OpenManus/config/mcp_servers.json", "r"
        ) as f:
            config = json.load(f)

        agent = MCPAgent()

        # 提取playwright服务的配置
        server_config = config["mcpServers"]["playwright"]
        command = server_config.get("command")
        args = server_config.get("args")

        # 初始化agent
        await agent.initialize(
            connection_type="stdio",
            command=command,
            args=args,
        )

        # 测试不同功能
        test_queries = [
            "打开百度搜索今日新闻，总结，然后终止任务",
        ]

        for query in test_queries:
            logger.info(f"正在查询: {query}")
            result = await agent.run(query)
            logger.info(f"查询结果: {result}")

    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        raise
    finally:
        if "agent" in locals():
            await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(test_playwright_service())
