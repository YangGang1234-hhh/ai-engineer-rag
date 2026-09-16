# Agent复杂编排模式总结

## 1. 串行编排（Serial Orchestration）

### 概念
串行编排是指将多个Agent按顺序依次执行，前一个Agent的输出作为后一个Agent的输入，形成一个线性的工作流。

### 应用场景
- 需要逐步处理的复杂任务
- 任务有明确的前后依赖关系
- 需要质量检查或验证的流程

### 实现方式
使用 `Runner.run()` 依次调用多个Agent，每个Agent的输出作为下一个Agent的输入。

### 示例代码（30_串行编排.py）

```python
# 1. 生成故事大纲
outline_result = await Runner.run(story_outline_agent, input_prompt)

# 2. 检查大纲质量
outline_checker_result = await Runner.run(outline_checker_agent, outline_result.final_output)

# 3. 质量门控
if not outline_checker_result.final_output.good_quality:
    print("故事不吸引人，停止输出...")
    exit(0)

# 4. 编写故事
story_result = await Runner.run(story_agent, outline_result.final_output)
```

### 优势
- 流程清晰，易于理解和调试
- 可以在每个步骤进行质量检查
- 便于添加门控逻辑

---

## 2. Handoffs路由机制

### 概念
Handoffs（交接）是一种动态路由机制，由一个主Agent（triage agent）根据用户输入的内容，自动将任务分配给最合适的专业Agent。

### 应用场景
- 多领域问题处理
- 客服系统
- 智能助手路由

### 实现方式
定义一个主Agent，通过 `handoffs` 参数指定可以交接的专业Agent列表，主Agent会根据用户意图自动选择合适的Agent。

### 示例代码（31_Handoffs路由.py）

```python
# 定义专业Agent
math_agent = Agent(
    name="math_agent",
    model="qwen-max",
    instructions="你是小王，擅长数学计算，回答问题的时候先告诉我你是谁。",
)

language_agent = Agent(
    name="language_agent",
    model="qwen-max",
    instructions="你是小李，擅长将翻译，回答问题的时候先告诉我你是谁。",
)

sport_agent = Agent(
    name="sport_agent",
    model="qwen-max",
    instructions="你是小张，擅长介绍各种体育运动，回答问题的时候先告诉我你是谁。",
)

# 定义路由Agent
triage_agent = Agent(
    name="triage_agent",
    model="qwen-max",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[math_agent, language_agent, sport_agent],
)
```

### 优势
- 自动识别用户意图
- 专业Agent处理专业问题
- 无缝的用户体验

---

## 2.1 Handoffs路由学习指南

### 第一部分：学习Handoffs路由机制（30分钟）

#### 学习目标
- 理解Handoffs路由的核心概念
- 掌握Handoffs的应用场景
- 了解Handoffs的实现原理

#### 需要掌握的核心代码

**1. 定义专业Agent（最基础）**
```python
# 每个专业Agent负责特定领域
math_agent = Agent(
    name="math_agent",           # Agent名称，用于识别
    model="qwen-max",             # 使用的模型
    instructions="你是小王，擅长数学计算，回答问题的时候先告诉我你是谁。",
)
```

**代码要点：**
- `name`：必须唯一，用于标识Agent
- `instructions`：明确告诉Agent自己的身份和职责
- 每个Agent专注于一个领域

---

### 第二部分：实现31_Handoffs路由示例（90分钟）

#### 学习目标
- 能够独立编写Handoffs路由代码
- 成功实现任务自动分配
- 理解流式输出的实现

#### 需要掌握的核心代码

**1. 定义路由Agent（核心代码）**
```python
# 路由Agent是核心，负责分发任务
triage_agent = Agent(
    name="triage_agent",
    model="qwen-max",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[math_agent, language_agent, sport_agent],  # 关键：指定可以交接的Agent列表
)
```

**代码要点：**
- `handoffs` 参数是Handoffs的核心
- 传入一个Agent列表，路由Agent可以选择其中任何一个
- 路由Agent的instructions要明确告诉它如何选择

**2. 流式输出实现（重要）**
```python
# 使用流式输出，让用户体验更好
result = Runner.run_streamed(
    agent,
    input=inputs,
)

async for event in result.stream_events():
    if isinstance(event.data, ResponseTextDeltaEvent):
        print(data.delta, end="", flush=True)  # 实时打印文本增量
```

**代码要点：**
- `Runner.run_streamed()` 用于流式输出
- 遍历 `stream_events()` 获取每个事件
- 判断 `ResponseTextDeltaEvent` 类型来获取文本增量
- 使用 `flush=True` 确保实时输出

**3. 保持对话上下文（关键）**
```python
# 将当前结果转换为输入列表，保持上下文
inputs = result.to_input_list()

# 添加新的用户消息
inputs.append({"content": user_msg, "role": "user"})

# 获取当前处理的Agent（可能已经交接）
agent = result.current_agent
```

**代码要点：**
- `to_input_list()` 保存对话历史
- 每次都要添加新的用户消息
- `current_agent` 获取当前正在处理的Agent（可能已交接）

---

### 第三部分：测试路由功能（30分钟）

#### 学习目标
- 成功测试Handoffs路由功能
- 验证不同任务类型是否能正确分配
- 能够排查常见问题

#### 测试用例建议

**1. 数学问题测试**
```
输入："计算 1 + 1 等于多少"
期望：math_agent 接手回答
验证点：回答中包含"我是小王"
```

**2. 翻译问题测试**
```
输入："把 '你好' 翻译成英文"
期望：language_agent 接手回答
验证点：回答中包含"我是小李"
```

**3. 运动问题测试**
```
输入："介绍一下篮球"
期望：sport_agent 接手回答
验证点：回答中包含"我是小张"
```

**4. 连续对话测试**
```
第一轮："计算 2 + 2" → math_agent
第二轮："再把结果翻译成英文" → language_agent
验证点：第二轮能自动切换到翻译Agent
```

#### 常见问题排查

**问题1：路由Agent不交接，自己回答**
```
原因：instructions写得不够明确
解决：明确告诉它"不要自己回答，一定要交接给其他Agent"
```

**问题2：交接错误的Agent**
```
原因：专业Agent的instructions或name不够清晰
解决：给每个Agent更明确的身份描述
```

**问题3：对话上下文丢失**
```
原因：没有正确使用 to_input_list()
解决：确保每次都调用 inputs = result.to_input_list()
```

---

## 3. Orchestrator路由设计

### 概念
Orchestrator（编排器）是一种集中式任务调度模式，将其他Agent作为工具调用，可以更灵活地控制任务流程，支持并行调用、顺序调用等复杂逻辑。

### 应用场景
- 复杂任务调度
- 多步骤工作流
- 需要精确控制执行顺序的场景

### 实现方式
将其他Agent通过 `as_tool()` 方法转换为工具，然后在Orchestrator Agent的 `tools` 参数中注册，Orchestrator可以根据需要调用这些工具。

### 示例代码（32_Orchestrator路由.py）

```python
# 将Agent转换为工具
orchestrator_agent = Agent(
    name="orchestrator_agent",
    model="qwen-max",
    instructions=(
        "You are math / language and sport agent. You use the tools given to you to response."
        "If asked for multiple task, you call the relevant tools in order."
        "You never translate on your own, you always use the provided tools."
    ),
    tools=[
        math_agent.as_tool(
            tool_name="slove_math",
            tool_description="解决数学题",
        ),
        language_agent.as_tool(
            tool_name="translate_language",
            tool_description="进行文本翻译",
        ),
        sport_agent.as_tool(
            tool_name="sport_introduction",
            tool_description="介绍运动行为",
        ),
    ],
)
```

### 优势
- 集中式任务调度
- 支持复杂的执行逻辑
- 可以同时调用多个工具
- 更灵活的控制能力

---

## 三种模式对比

| 特性 | 串行编排 | Handoffs路由 | Orchestrator路由 |
|------|---------|-------------|-----------------|
| 执行方式 | 线性顺序 | 动态选择 | 集中式调度 |
| 控制粒度 | 步骤级 | Agent级 | 工具级 |
| 灵活性 | 低 | 中 | 高 |
| 复杂度 | 简单 | 中等 | 复杂 |
| 适用场景 | 线性任务 | 多领域路由 | 复杂工作流 |

---

## 实际应用建议

1. **简单线性任务** → 使用串行编排
2. **多领域问题处理** → 使用Handoffs路由
3. **复杂任务调度** → 使用Orchestrator路由

可以根据实际需求灵活选择或组合使用这些模式！
