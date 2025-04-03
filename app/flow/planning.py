import asyncio
import json
import os
import time
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import Field

from app.agent.base import BaseAgent
from app.flow.base import BaseFlow
from app.llm import LLM
from app.logger import logger
from app.schema import AgentState, Message, ToolChoice
from app.tool import PlanningTool


class PlanStepStatus(str, Enum):
    """Enum class defining possible statuses of a plan step"""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"

    @classmethod
    def get_all_statuses(cls) -> list[str]:
        """Return a list of all possible step status values"""
        return [status.value for status in cls]

    @classmethod
    def get_active_statuses(cls) -> list[str]:
        """Return a list of values representing active statuses (not started or in progress)"""
        return [cls.NOT_STARTED.value, cls.IN_PROGRESS.value]

    @classmethod
    def get_status_marks(cls) -> Dict[str, str]:
        """Return a mapping of statuses to their marker symbols"""
        return {
            cls.COMPLETED.value: "[✓]",
            cls.IN_PROGRESS.value: "[→]",
            cls.BLOCKED.value: "[!]",
            cls.NOT_STARTED.value: "[ ]",
        }


class PlanningFlow(BaseFlow):
    """A flow that manages planning and execution of tasks using agents."""

    llm: LLM = Field(default_factory=lambda: LLM())
    planning_tool: PlanningTool = Field(default_factory=PlanningTool)
    executor_keys: List[str] = Field(default_factory=list)
    active_plan_id: str = Field(default_factory=lambda: f"plan_{int(time.time())}")
    current_step_index: Optional[int] = None

    def __init__(
        self, agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]], **data
    ):
        # Set executor keys before super().__init__
        if "executors" in data:
            data["executor_keys"] = data.pop("executors")

        # Set plan ID if provided
        if "plan_id" in data:
            data["active_plan_id"] = data.pop("plan_id")

        # Initialize the planning tool if not provided
        if "planning_tool" not in data:
            planning_tool = PlanningTool()
            data["planning_tool"] = planning_tool

        # Call parent's init with the processed data
        super().__init__(agents, **data)

        # Set executor_keys to all agent keys if not specified
        if not self.executor_keys:
            self.executor_keys = list(self.agents.keys())

    def get_executor(self, step_type: Optional[str] = None) -> BaseAgent:
        """
        Get an appropriate executor agent for the current step.
        Selects an agent based on step_type if provided and available,
        otherwise falls back to the first available executor or the primary agent.

        Args:
            step_type: The type/category extracted from the plan step (e.g., 'amap_maps', 'browser_use').

        Returns:
            The selected BaseAgent instance.
        """
        selected_agent_key = None

        # Try to select agent based on step_type
        if step_type:
            # 记录原始步骤类型，用于日志
            original_step_type = step_type

            # 尝试直接匹配
            if step_type in self.agents:
                selected_agent_key = step_type
                logger.info(
                    f"Selecting executor agent based on step_type: '{step_type}'"
                )
                return self.agents[step_type]

            # 尝试将下划线替换为连字符，以匹配可能的agent key格式
            normalized_step_type = step_type.replace("_", "-")
            if normalized_step_type in self.agents:
                selected_agent_key = normalized_step_type
                logger.info(
                    f"Selecting executor agent based on normalized step_type: '{normalized_step_type}'"
                )
                return self.agents[normalized_step_type]

            # 尝试将连字符替换为下划线，以匹配可能的agent key格式
            normalized_step_type = step_type.replace("-", "_")
            if normalized_step_type in self.agents:
                selected_agent_key = normalized_step_type
                logger.info(
                    f"Selecting executor agent based on normalized step_type: '{normalized_step_type}'"
                )
                return self.agents[normalized_step_type]

            # 尝试将步骤类型转换为小写，以匹配可能的agent key格式
            lowercase_step_type = step_type.lower()
            if lowercase_step_type in self.agents:
                selected_agent_key = lowercase_step_type
                logger.info(
                    f"Selecting executor agent based on lowercase step_type: '{lowercase_step_type}'"
                )
                return self.agents[lowercase_step_type]

            # 尝试部分匹配 - 检查步骤类型是否是任何agent key的子字符串
            for agent_key in self.agents.keys():
                if (
                    lowercase_step_type in agent_key.lower()
                    or agent_key.lower() in lowercase_step_type
                ):
                    selected_agent_key = agent_key
                    logger.info(
                        f"Selecting executor agent based on partial match: '{agent_key}' for step_type '{step_type}'"
                    )
                    return self.agents[agent_key]

            # 记录未匹配的步骤类型
            logger.warning(
                f"Step type '{original_step_type}' could not be matched to any agent. Available agents: {list(self.agents.keys())}"
            )

        # 如果没有步骤类型或无法匹配，尝试从executor_keys中选择最合适的执行器
        # 优先选择非主要agent的执行器，因为主要agent通常是通用的
        non_primary_executors = [
            key for key in self.executor_keys if key != self.primary_agent_key
        ]
        if non_primary_executors:
            for key in non_primary_executors:
                if key in self.agents:
                    selected_agent_key = key
                    logger.info(
                        f"Step_type '{step_type}' not found or not specified. Selecting non-primary executor: '{key}'"
                    )
                    return self.agents[key]

        # 如果没有非主要执行器，则使用主要agent
        selected_agent_key = self.primary_agent_key
        logger.info(
            f"Step_type '{step_type}' not found or not specified. Falling back to primary agent: '{selected_agent_key}'"
        )
        return self.primary_agent

    async def execute(self, input_text: str) -> str:
        """Execute the planning flow with agents."""
        try:
            if not self.primary_agent:
                raise ValueError("No primary agent available")

            # Create initial plan if input provided
            if input_text:
                await self._create_initial_plan(input_text)

                # Verify plan was created successfully
                if self.active_plan_id not in self.planning_tool.plans:
                    logger.error(
                        f"Plan creation failed. Plan ID {self.active_plan_id} not found in planning tool."
                    )
                    return f"Failed to create plan for: {input_text}"

            result = ""
            while True:
                # Get current step to execute
                self.current_step_index, step_info = await self._get_current_step_info()

                # Exit if no more steps or plan completed
                if self.current_step_index is None:
                    result += await self._finalize_plan()
                    break

                # Execute current step with appropriate agent
                step_type = step_info.get("type") if step_info else None
                executor = self.get_executor(step_type)
                step_result = await self._execute_step(executor, step_info)
                result += step_result + "\n"

                # Check if agent wants to terminate
                if hasattr(executor, "state") and executor.state == AgentState.FINISHED:
                    break

            return result
        except Exception as e:
            logger.error(f"Error in PlanningFlow: {str(e)}")
            return f"Execution failed: {str(e)}"

    async def _create_initial_plan(self, request: str) -> None:
        """Create an initial plan based on the request using the flow's LLM and PlanningTool."""
        logger.info(f"Creating initial plan with ID: {self.active_plan_id}")

        # --- Start modification: Build dynamic system prompt with agent info ---
        agent_descriptions = []
        # Iterate through executor keys to list specialized agents
        for agent_key in self.executor_keys:
            # Skip the primary agent if it's listed as an executor,
            # assuming it handles general tasks unless explicitly tagged otherwise.
            if agent_key == self.primary_agent_key:
                continue

            # Generate tag and basic capability description from the agent key
            tag = f"[{agent_key.upper()}]"  # e.g., [AMAP_MAPS]
            # Attempt to get a better description from the agent instance if possible
            capability_desc = f"tasks related to {agent_key.replace('_', ' ')}"
            if agent_key in self.agents and hasattr(
                self.agents[agent_key], "description"
            ):
                # Use agent's description if available, otherwise use the generated one
                agent_desc_attr = getattr(self.agents[agent_key], "description", None)
                if agent_desc_attr:
                    capability_desc = agent_desc_attr

            agent_descriptions.append(
                f"- For {capability_desc}, prefix the step with the tag {tag}. Example: '{tag} Perform {agent_key} task.'"
            )

        available_agents_info = ""
        if agent_descriptions:
            available_agents_info = (
                "\n\nAvailable specialized agents and their tags:\n"
                + "\n".join(agent_descriptions)
            )
            available_agents_info += "\nFor general tasks or steps without a specific agent requirement, do not add a tag."

        # Base system prompt
        base_system_prompt = (
            "You are a planning assistant. Create a concise, actionable plan with clear steps. "
            "Focus on key milestones rather than detailed sub-steps. "
            "Optimize for clarity and efficiency."
            "\n\nCRITICAL INSTRUCTION: For each step that requires specialized tools, you MUST prefix the step with the appropriate agent tag as listed below. For example, if a step requires maps functionality, prefix it with [AMAP-MAPS]. If a step requires browser functionality, prefix it with [PLAYWRIGHT]. This is ABSOLUTELY REQUIRED for the system to select the correct agent for each step.\n\nEXAMPLE PLAN WITH CORRECT TAGS:\n0. [ ] Analyze user request\n1. [ ] [AMAP-MAPS] Find location coordinates\n2. [ ] [AMAP-MAPS] Calculate driving route\n3. [ ] [PLAYWRIGHT] Search for additional information\n4. [ ] [AMAP-MAPS] Generate map visualization\n5. [ ] Summarize results\n\nNOTE: Without proper tags, steps will default to the general agent and may not execute correctly. EVERY specialized step MUST have a tag."
            f"{available_agents_info}"  # Append the dynamic agent info
            "\n\nIMPORTANT: You MUST use the 'planning' tool to create the plan. DO NOT just write out the plan as text. Instead, use the planning tool with command='create', providing a title and steps array."
            '\n\nEXAMPLE OF CORRECT TOOL USAGE:\n```\n{\n  "command": "create",\n  "title": "Plan title here",\n  "steps": [\n    "Step 1: Analyze request",\n    "Step 2: [PLAYWRIGHT] Search for information",\n    "Step 3: Summarize findings"\n  ]\n}\n```\nYou MUST format your response as a tool call like the example above.'
        )

        # Create the final system message
        system_message = Message.system_message(base_system_prompt)
        # --- End modification ---

        # Create a user message with the request
        user_message = Message.user_message(
            f"Create a reasonable plan with clear steps to accomplish the task: {request}"
        )

        # 添加明确的指导，要求LLM使用工具调用格式返回结果
        user_message = Message.user_message(
            f"Create a reasonable plan with clear steps to accomplish the task: {request}\n\n"
            + "IMPORTANT: You MUST use the 'planning' tool to create the plan. Do not just describe the plan in text. "
            + "Use the tool with command='create', providing a title and steps array."
        )

        # Call LLM with PlanningTool, using the dynamically generated system message
        response = await self.llm.ask_tool(
            messages=[user_message],
            system_msgs=[system_message],  # Use the updated system message
            tools=[self.planning_tool.to_param()],
            tool_choice=ToolChoice.AUTO,
        )

        # 记录LLM的完整响应内容，以便调试
        logger.info(f"LLM响应内容: {response}")

        # Process tool calls if present
        if response and response.tool_calls:
            logger.info(f"检测到工具调用数量: {len(response.tool_calls)}")
            for i, tool_call in enumerate(response.tool_calls):
                logger.info(f"处理工具调用 #{i+1}: {tool_call.function.name}")

                if tool_call.function.name == "planning":
                    # 记录原始参数
                    raw_args = tool_call.function.arguments
                    logger.info(f"原始工具参数: {raw_args}")

                    # Parse the arguments
                    args = raw_args
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                            logger.info(f"解析后的参数: {args}")
                        except json.JSONDecodeError as e:
                            logger.error(f"解析工具参数失败: {e}")
                            logger.error(f"问题参数内容: {args}")
                            continue

                    # Ensure plan_id is set correctly and execute the tool
                    args["plan_id"] = self.active_plan_id
                    logger.info(f"执行planning工具，计划ID: {self.active_plan_id}")

                    try:
                        # Execute the tool via ToolCollection instead of directly
                        result = await self.planning_tool.execute(**args)
                        logger.info(f"计划创建结果: {str(result)}")
                        return
                    except Exception as e:
                        logger.error(f"执行planning工具时出错: {str(e)}")
                        # 继续执行，创建默认计划
                else:
                    logger.warning(f"收到非planning工具调用: {tool_call.function.name}")
        else:
            logger.warning("LLM响应中没有工具调用")
            if response:
                logger.info(f"LLM响应内容类型: {type(response)}")
                logger.info(f"LLM响应属性: {dir(response)}")
                if hasattr(response, "content") and response.content:
                    logger.info(f"LLM响应文本内容: {response.content}")
                    # 尝试从content字段解析JSON格式的计划数据
                    try:
                        content_json = json.loads(response.content)
                        # 检查是否包含计划所需的关键字段
                        if (
                            isinstance(content_json, dict)
                            and "command" in content_json
                            and "steps" in content_json
                        ):
                            logger.info(
                                f"从content字段成功解析到计划数据: {content_json}"
                            )
                            # 添加计划ID
                            content_json["plan_id"] = self.active_plan_id
                            # 执行planning工具创建计划
                            try:
                                result = await self.planning_tool.execute(
                                    **content_json
                                )
                                logger.info(f"从content字段创建计划结果: {result}")
                                return
                            except Exception as e:
                                logger.error(
                                    f"从content字段执行planning工具时出错: {str(e)}"
                                )
                                # 继续执行，创建默认计划
                    except json.JSONDecodeError:
                        logger.warning(
                            "content字段内容不是有效的JSON格式，无法解析为计划数据"
                        )

        # If execution reached here, create a default plan
        logger.warning("创建默认计划 - LLM工具调用失败或未返回有效的planning工具调用")
        logger.info(f"请求内容: {request}")

        # 记录默认计划详情
        default_plan = {
            "command": "create",
            "plan_id": self.active_plan_id,
            "title": f"Plan for: {request[:50]}{'...' if len(request) > 50 else ''}",
            "steps": ["Analyze request", "Execute task", "Verify results"],
        }
        logger.info(f"默认计划详情: {default_plan}")

        try:
            # Create default plan using the ToolCollection
            result = await self.planning_tool.execute(**default_plan)
            logger.info(f"默认计划创建结果: {result}")
        except Exception as e:
            logger.error(f"创建默认计划时出错: {str(e)}")
            # 记录更多调试信息
            logger.error(f"计划工具状态: {self.planning_tool}")
            logger.error(
                f"现有计划: {list(self.planning_tool.plans.keys()) if hasattr(self.planning_tool, 'plans') else '无计划数据'}"
            )

    async def _get_current_step_info(self) -> tuple[Optional[int], Optional[dict]]:
        """
        Parse the current plan to identify the first non-completed step's index and info.
        Extracts step type (e.g., 'amap_maps') from step text like '[AMAP_MAPS] Find route'.
        Returns (None, None) if no active step is found
        """
        if (
            not self.active_plan_id
            or self.active_plan_id not in self.planning_tool.plans
        ):
            logger.error(f"Plan with ID {self.active_plan_id} not found")
            return None, None

        try:
            # Direct access to plan data from planning tool storage
            plan_data = self.planning_tool.plans[self.active_plan_id]
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])

            # Find first non-completed step
            for i, step in enumerate(steps):
                if i >= len(step_statuses):
                    status = PlanStepStatus.NOT_STARTED.value
                else:
                    status = step_statuses[i]

                if status in PlanStepStatus.get_active_statuses():
                    # Extract step type/category if available
                    step_info = {"text": step}

                    # Try to extract step type from the text (e.g., [SEARCH], [CODE], [AMAP-MAPS])
                    import re

                    # 支持带连字符的标签格式，如[AMAP-MAPS]，并且不区分大小写
                    type_match = re.search(r"\[([A-Za-z_-]+)\]", step)
                    if type_match:
                        # 提取步骤类型并保留原始格式
                        original_step_type = type_match.group(1)
                        # 同时存储原始格式和小写格式，以便于匹配
                        step_info["type"] = original_step_type
                        step_info["type_lower"] = original_step_type.lower()
                        logger.info(
                            f"Extracted step type: '{original_step_type}' from step: '{step}'"
                        )
                    else:
                        logger.warning(f"No step type tag found in step: '{step}'")
                        # 默认使用None，但在get_executor方法中会处理这种情况
                        step_info["type"] = None

                        # 尝试从步骤文本中提取关键词来推断步骤类型
                        step_lower = step.lower()
                        for agent_key in self.executor_keys:
                            # 检查步骤文本是否包含agent_key（忽略大小写）
                            if agent_key.lower() in step_lower:
                                step_info["type"] = agent_key
                                step_info["type_lower"] = agent_key.lower()
                                logger.info(
                                    f"Inferred step type: '{agent_key}' from step text: '{step}'"
                                )
                                break

                    # Mark current step as in_progress
                    try:
                        await self.planning_tool.execute(
                            command="mark_step",
                            plan_id=self.active_plan_id,
                            step_index=i,
                            step_status=PlanStepStatus.IN_PROGRESS.value,
                        )
                    except Exception as e:
                        logger.warning(f"Error marking step as in_progress: {e}")
                        # Update step status directly if needed
                        if i < len(step_statuses):
                            step_statuses[i] = PlanStepStatus.IN_PROGRESS.value
                        else:
                            while len(step_statuses) < i:
                                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
                            step_statuses.append(PlanStepStatus.IN_PROGRESS.value)

                        plan_data["step_statuses"] = step_statuses

                    return i, step_info

            return None, None  # No active step found

        except Exception as e:
            logger.warning(f"Error finding current step index: {e}")
            return None, None

    async def _execute_step(self, executor: BaseAgent, step_info: dict) -> str:
        """Execute the current step with the specified agent using agent.run()."""
        # Prepare context for the agent with current plan status
        plan_status = await self._get_plan_text()
        step_text = step_info.get("text", f"Step {self.current_step_index}")

        # 添加步骤类型信息到提示中，帮助agent理解当前任务类型
        step_type_info = ""
        if "type" in step_info and step_info["type"]:
            step_type_info = f"\n\n步骤类型: {step_info['type']}"
            logger.info(
                f"Executing step with type: {step_info['type']} using agent: {executor.name}"
            )
        else:
            logger.info(
                f"Executing step without specific type using agent: {executor.name}"
            )

        # 如果执行器是MCPAgent，在执行步骤前重新初始化它
        is_mcp_agent = False
        from app.agent.mcp import MCPAgent

        if isinstance(executor, MCPAgent):
            is_mcp_agent = True
            agent_key = None
            # 找出executor对应的key
            for key, agent in self.agents.items():
                if agent is executor:
                    agent_key = key
                    break

            logger.info(
                f"Preparing to initialize MCPAgent for step {self.current_step_index}: {agent_key}"
            )

            # 使用预初始化时存储的配置信息
            try:
                # 检查是否有存储的服务器配置
                if hasattr(executor, "server_config") and executor.server_config:
                    # 使用存储的配置信息
                    server_config = executor.server_config

                    # 设置环境变量
                    if "env" in server_config and server_config["env"]:
                        for key, value in server_config["env"].items():
                            os.environ[key] = value

                    # 初始化agent（异步操作）
                    logger.info(
                        f"Initializing MCPAgent for step {self.current_step_index}: {agent_key} using stored config"
                    )
                    await executor.initialize(
                        connection_type=server_config.get("connection_type", "stdio"),
                        command=server_config.get("command"),
                        args=server_config.get("args", []),
                    )
                    logger.info(
                        f"Successfully initialized MCPAgent for step {self.current_step_index}: {agent_key}"
                    )
                else:
                    # 如果没有存储的配置，从配置文件加载
                    import json

                    # 使用全局导入的os模块，不要在局部作用域重新导入

                    config_path = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "config",
                        "mcp_servers.json",
                    )
                    if os.path.exists(config_path):
                        with open(config_path, "r") as f:
                            config = json.load(f)

                        # 找到对应的MCP服务器配置
                        if agent_key in config.get("mcpServers", {}):
                            server_config = config["mcpServers"][agent_key]

                            # 设置环境变量
                            if "env" in server_config:
                                for key, value in server_config["env"].items():
                                    os.environ[key] = value

                            # 初始化agent（异步操作）
                            logger.info(
                                f"Initializing MCPAgent for step {self.current_step_index}: {agent_key} using config file"
                            )
                            await executor.initialize(
                                connection_type="stdio",
                                command=server_config.get("command"),
                                args=server_config.get("args", []),
                            )
                            logger.info(
                                f"Successfully initialized MCPAgent for step {self.current_step_index}: {agent_key}"
                            )
            except Exception as e:
                logger.error(
                    f"Failed to initialize MCPAgent for step {self.current_step_index}: {str(e)}"
                )

        # Create a prompt for the agent to execute the current step
        step_prompt = f"""
        CURRENT PLAN STATUS:
        {plan_status}

        YOUR CURRENT TASK:
        You are now working on step {self.current_step_index}: "{step_text}"{step_type_info}

        Please execute this step using the appropriate tools. When you're done, provide a summary of what you accomplished.
        """

        # Use agent.run() to execute the step
        try:
            step_result = await executor.run(step_prompt)

            # Mark the step as completed after successful execution
            await self._mark_step_completed()

            # 如果执行器是MCPAgent，在步骤执行后立即清理它
            if is_mcp_agent:
                logger.info(
                    f"Cleaning up MCPAgent after step {self.current_step_index}"
                )
                try:
                    # 使用超时机制避免清理过程阻塞
                    await asyncio.wait_for(executor.cleanup(), timeout=10.0)
                    logger.info(
                        f"Successfully cleaned up MCPAgent after step {self.current_step_index}"
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Cleanup for MCPAgent timed out after 10 seconds")
                    # 强制清理资源
                    if hasattr(executor, "mcp_clients"):
                        executor.mcp_clients.session = None
                        executor.mcp_clients.tools = tuple()
                        executor.mcp_clients.tool_map = {}
                    logger.info(
                        f"Forced resource release for MCPAgent after step {self.current_step_index}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error cleaning up MCPAgent after step {self.current_step_index}: {str(e)}"
                    )
                    # 强制清理资源
                    if hasattr(executor, "mcp_clients"):
                        executor.mcp_clients.session = None
                        executor.mcp_clients.tools = tuple()
                        executor.mcp_clients.tool_map = {}
                    logger.info(
                        f"Forced resource release for MCPAgent after step {self.current_step_index}"
                    )

            return step_result
        except Exception as e:
            logger.error(f"Error executing step {self.current_step_index}: {e}")

            # 如果执行器是MCPAgent，在步骤执行失败后也要清理它
            if is_mcp_agent:
                logger.info(
                    f"Cleaning up MCPAgent after step {self.current_step_index} failed"
                )
                try:
                    # 使用超时机制避免清理过程阻塞
                    await asyncio.wait_for(executor.cleanup(), timeout=10.0)
                    logger.info(
                        f"Successfully cleaned up MCPAgent after step {self.current_step_index} failed"
                    )
                except Exception as e2:
                    logger.error(
                        f"Error cleaning up MCPAgent after step {self.current_step_index} failed: {str(e2)}"
                    )
                    # 强制清理资源
                    if hasattr(executor, "mcp_clients"):
                        executor.mcp_clients.session = None
                        executor.mcp_clients.tools = tuple()
                        executor.mcp_clients.tool_map = {}
                    logger.info(
                        f"Forced resource release for MCPAgent after step {self.current_step_index} failed"
                    )

            return f"Error executing step {self.current_step_index}: {str(e)}"

    async def _mark_step_completed(self) -> None:
        """Mark the current step as completed."""
        if self.current_step_index is None:
            return

        try:
            # Mark the step as completed
            await self.planning_tool.execute(
                command="mark_step",
                plan_id=self.active_plan_id,
                step_index=self.current_step_index,
                step_status=PlanStepStatus.COMPLETED.value,
            )
            logger.info(
                f"Marked step {self.current_step_index} as completed in plan {self.active_plan_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to update plan status: {e}")
            # Update step status directly in planning tool storage
            if self.active_plan_id in self.planning_tool.plans:
                plan_data = self.planning_tool.plans[self.active_plan_id]
                step_statuses = plan_data.get("step_statuses", [])

                # Ensure the step_statuses list is long enough
                while len(step_statuses) <= self.current_step_index:
                    step_statuses.append(PlanStepStatus.NOT_STARTED.value)

                # Update the status
                step_statuses[self.current_step_index] = PlanStepStatus.COMPLETED.value
                plan_data["step_statuses"] = step_statuses

    async def _get_plan_text(self) -> str:
        """Get the current plan as formatted text."""
        try:
            result = await self.planning_tool.execute(
                command="get", plan_id=self.active_plan_id
            )
            return result.output if hasattr(result, "output") else str(result)
        except Exception as e:
            logger.error(f"Error getting plan: {e}")
            return self._generate_plan_text_from_storage()

    def _generate_plan_text_from_storage(self) -> str:
        """Generate plan text directly from storage if the planning tool fails."""
        try:
            if self.active_plan_id not in self.planning_tool.plans:
                return f"Error: Plan with ID {self.active_plan_id} not found"

            plan_data = self.planning_tool.plans[self.active_plan_id]
            title = plan_data.get("title", "Untitled Plan")
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])
            step_notes = plan_data.get("step_notes", [])

            # Ensure step_statuses and step_notes match the number of steps
            while len(step_statuses) < len(steps):
                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
            while len(step_notes) < len(steps):
                step_notes.append("")

            # Count steps by status
            status_counts = {status: 0 for status in PlanStepStatus.get_all_statuses()}

            for status in step_statuses:
                if status in status_counts:
                    status_counts[status] += 1

            completed = status_counts[PlanStepStatus.COMPLETED.value]
            total = len(steps)
            progress = (completed / total) * 100 if total > 0 else 0

            plan_text = f"Plan: {title} (ID: {self.active_plan_id})\n"
            plan_text += "=" * len(plan_text) + "\n\n"

            plan_text += (
                f"Progress: {completed}/{total} steps completed ({progress:.1f}%)\n"
            )
            plan_text += f"Status: {status_counts[PlanStepStatus.COMPLETED.value]} completed, {status_counts[PlanStepStatus.IN_PROGRESS.value]} in progress, "
            plan_text += f"{status_counts[PlanStepStatus.BLOCKED.value]} blocked, {status_counts[PlanStepStatus.NOT_STARTED.value]} not started\n\n"
            plan_text += "Steps:\n"

            status_marks = PlanStepStatus.get_status_marks()

            for i, (step, status, notes) in enumerate(
                zip(steps, step_statuses, step_notes)
            ):
                # Use status marks to indicate step status
                status_mark = status_marks.get(
                    status, status_marks[PlanStepStatus.NOT_STARTED.value]
                )

                plan_text += f"{i}. {status_mark} {step}\n"
                if notes:
                    plan_text += f"   Notes: {notes}\n"

            return plan_text
        except Exception as e:
            logger.error(f"Error generating plan text from storage: {e}")
            return f"Error: Unable to retrieve plan with ID {self.active_plan_id}"

    async def _finalize_plan(self) -> str:
        """Finalize the plan and provide a summary using the flow's LLM directly."""
        plan_text = await self._get_plan_text()

        # Create a summary using the flow's LLM directly
        try:
            system_message = Message.system_message(
                "You are a planning assistant. Your task is to summarize the completed plan."
            )

            user_message = Message.user_message(
                f"The plan has been completed. Here is the final plan status:\n\n{plan_text}\n\nPlease provide a summary of what was accomplished and any final thoughts."
            )

            response = await self.llm.ask(
                messages=[user_message], system_msgs=[system_message]
            )

            return f"Plan completed:\n\n{response}"
        except Exception as e:
            logger.error(f"Error finalizing plan with LLM: {e}")

            # Fallback to using an agent for the summary
            try:
                agent = self.primary_agent
                summary_prompt = f"""
                The plan has been completed. Here is the final plan status:

                {plan_text}

                Please provide a summary of what was accomplished and any final thoughts.
                """
                summary = await agent.run(summary_prompt)
                return f"Plan completed:\n\n{summary}"
            except Exception as e2:
                logger.error(f"Error finalizing plan with agent: {e2}")
                return "Plan completed. Error generating summary."
