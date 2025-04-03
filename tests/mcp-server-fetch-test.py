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


async def test_fetch_service():
    """测试fetch MCP服务"""
    try:
        # 读取配置文件
        import json

        with open(
            "/Users/huangweihao/openmanus/OpenManus/config/mcp_servers.json", "r"
        ) as f:
            config = json.load(f)

        agent = MCPAgent()

        # 提取fetch服务的配置
        server_config = config["mcpServers"]["fetch"]
        command = server_config.get("command")
        args = server_config.get("args")

        # 初始化agent
        await agent.initialize(
            connection_type="stdio",
            command=command,
            args=args,
        )

        # 测试查询
        test_queries = [
            "获取'https://news.sina.com.cn'内容并输出，然后终止任务",
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
            try:
                await agent.cleanup()
            except Exception as e:
                logger.error(f"清理agent时发生错误: {str(e)}")


if __name__ == "__main__":
    asyncio.run(test_fetch_service())
