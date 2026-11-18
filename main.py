# TODO: improve CodeGenerator to align more with CodeReviewer standards to prevent infinite back&forth.
# TODO-maybe: maybe also add architecture spec to FEEDBACK loop to not lose context of the original spec.

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import os
from autogen_core.models import SystemMessage, UserMessage, AssistantMessage
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from autogen_core import (
    RoutedAgent,
    default_subscription,
    message_handler,
    MessageContext,
    AgentId,
    SingleThreadedAgentRuntime,
    DefaultTopicId,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient

#Env vars
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")

print(f"AutoGen and dependencies are installed successfully!")

# ==========================================
# 0. COMMON LLM CONFIGURATION
# ==========================================
DEFAULT_LLM_CONFIG = {
    "config_list": [
        {
            "model": "gpt-4o",
            "api_key": OPENAI_KEY,
            "base_url": OPENAI_ENDPOINT,
        }
    ],
    "temperature": 0.2,
    "timeout": 120,
}

# ==========================================
# 1. MESSAGE PROTOCOL WITH METADATA
# ==========================================
@dataclass
class PipelineMessage:
    payload: str
    message_type: str
    sender_id: str
    receiver_id: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    # Custom protocol metadata fields
    trace_id: str = ""
    current_stage: str = ""
    target_stage: str = ""
    priority: int = 1
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


# ==========================================
# 2. DEFINE AGENTS (Using autogen-core 0.7.5)
# ==========================================

@default_subscription
class ClientAgent(RoutedAgent):
    # ClientAgent intentionally left without llm_config as requested
    def __init__(self, description: str) -> None:
        super().__init__(description)

    @message_handler
    async def handle_broadcast(self, message: PipelineMessage, ctx: MessageContext) -> None:
        print(f"\n[Client Agent] Received broadcast notification.")
        print(f" ├── Protocol Meta (Type): {message.message_type}")
        print(f" ├── Protocol Meta (Sender): {message.sender_id}")
        print(f" └── Notification (Payload):\n{message.payload}")

        # Send acknowledgment back to Reviewer Agent
        ack_msg = PipelineMessage(
            payload="Acknowledgment: Success broadcast received by Client.",
            message_type="ACKNOWLEDGMENT",
            sender_id="ClientAgent",
            receiver_id="ReviewerAgent",
            trace_id=message.trace_id,
            current_stage="complete",
            target_stage="code_review"
        )
        reviewer_id = AgentId("ReviewerAgent", "default")
        await self.send_message(ack_msg, reviewer_id)

@default_subscription
class SystemArchitectAgent(RoutedAgent):
    def __init__(self, description: str, llm_config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(description)
        self.llm_config = llm_config or DEFAULT_LLM_CONFIG
        config_entry = self.llm_config["config_list"][0]
        self.model_client = OpenAIChatCompletionClient(
            model=config_entry["model"],
            api_key=config_entry["api_key"],
            base_url=config_entry["base_url"]
        )

    @message_handler
    async def handle_task(self, message: PipelineMessage, ctx: MessageContext) -> None:
        if message.message_type == "FEATURE_REQUEST":
            print(f"\n[Architect Agent] Received message (Using model: {self.llm_config['config_list'][0]['model']}).")
            print(f" ├── Protocol Meta (Type): {message.message_type}")
            print(f" ├── Protocol Meta (Sender): {message.sender_id} -> Receiver: {message.receiver_id}")
            print(f" └── Payload: {message.payload}")

            messages = [
                SystemMessage(content="You are a senior software architect. Design the system architecture without giving specific implementation details like code snippets or file paths."),
                UserMessage(content=message.payload, source=message.sender_id)
            ]

            response = await self.model_client.create(
                messages=messages,
                cancellation_token=ctx.cancellation_token
            )

            print(f"[Architect Agent] LLM Response: {response.content}")

            response_msg = PipelineMessage(
                payload=f"SPEC: {response.content}",
                message_type="ARCHITECTURE_SPEC",
                sender_id="ArchitectAgent",
                receiver_id="CoderAgent",
                trace_id=message.trace_id,
                current_stage="architecture",
                target_stage="coding",
                priority=message.priority,
                extra_metadata={"schema_version": "1.0"}
            )
            coder_id = AgentId("CoderAgent", "default")
            await self.send_message(response_msg, coder_id)
        elif message.message_type == "SUCCESS_BROADCAST":
            print(f"\n[Architect Agent] Received success broadcast. Sending ACK...")
            ack_msg = PipelineMessage(
                payload="Acknowledgment: Success broadcast received by Architect.",
                message_type="ACKNOWLEDGMENT",
                sender_id="ArchitectAgent",
                receiver_id="ReviewerAgent",
                trace_id=message.trace_id,
                current_stage="complete",
                target_stage="code_review"
            )
            reviewer_id = AgentId("ReviewerAgent", "default")
            await self.send_message(ack_msg, reviewer_id)

@default_subscription
class BackendCoderAgent(RoutedAgent):
    def __init__(self, description: str, llm_config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(description)
        self.llm_config = llm_config or DEFAULT_LLM_CONFIG
        config_entry = self.llm_config["config_list"][0]
        self.model_client = OpenAIChatCompletionClient(
            model=config_entry["model"],
            api_key=config_entry["api_key"],
            base_url=config_entry["base_url"]
        )

    @message_handler
    async def handle_task(self, message: PipelineMessage, ctx: MessageContext) -> None:
        if message.message_type in ["ARCHITECTURE_SPEC", "CODE_FEEDBACK"]:
            print(f"\n[Coder Agent] Received message type: {message.message_type} (Using model: {self.llm_config['config_list'][0]['model']}).")
            print(f" ├── Protocol Meta (Sender): {message.sender_id} -> Receiver: {message.receiver_id}")
            print(f" └── Payload Received:\n{message.payload}")

            review_count = message.extra_metadata.get("review_count", 0)
            print(f"[Coder Agent] Generating/Adjusting code via AI model (Iteration: {review_count + 1})...")

            if message.message_type == "ARCHITECTURE_SPEC":
                prompt_content = f"Here is the architecture specification:\n{message.payload}"
            else:
                prompt_content = f"Previous code had issues. Reviewer feedback:\n{message.payload}\nPlease correct the code accordingly."

            messages = [
                SystemMessage(
                    content=(
                        "You are an expert backend software engineer. "
                        "Write clean, executable Python Flask code implementing the requirements. "
                        "Return only the raw code or markdown code block."
                    )
                ),
                UserMessage(content=prompt_content, source=message.sender_id)
            ]

            llm_response = await self.model_client.create(
                messages=messages,
                cancellation_token=ctx.cancellation_token
            )
            
            code_payload = llm_response.content
            print(f" └── [Coder Agent] Code generated/adjusted successfully by LLM.")

            response_msg = PipelineMessage(
                payload=code_payload,
                message_type="IMPLEMENTATION_CODE",
                sender_id="CoderAgent",
                receiver_id="ReviewerAgent",
                trace_id=message.trace_id,
                current_stage="coding",
                target_stage="code_review",
                priority=message.priority,
                extra_metadata={"review_count": review_count}
            )
            reviewer_id = AgentId("ReviewerAgent", "default")
            await self.send_message(response_msg, reviewer_id)
            
        elif message.message_type == "SUCCESS_BROADCAST":
            print(f"\n[Coder Agent] Received success broadcast. Sending ACK...")
            ack_msg = PipelineMessage(
                payload="Acknowledgment: Success broadcast received by Coder.",
                message_type="ACKNOWLEDGMENT",
                sender_id="CoderAgent",
                receiver_id="ReviewerAgent",
                trace_id=message.trace_id,
                current_stage="complete",
                target_stage="code_review"
            )
            reviewer_id = AgentId("ReviewerAgent", "default")
            await self.send_message(ack_msg, reviewer_id)

class CodeReviewerAgent(RoutedAgent):
    def __init__(self, description: str, llm_config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(description)
        self.llm_config = llm_config or DEFAULT_LLM_CONFIG
        config_entry = self.llm_config["config_list"][0]
        self.model_client = OpenAIChatCompletionClient(
            model=config_entry["model"],
            api_key=config_entry["api_key"],
            base_url=config_entry["base_url"]
        )

    @message_handler
    async def handle_task(self, message: PipelineMessage, ctx: MessageContext) -> None:
        if message.message_type == "IMPLEMENTATION_CODE":
            review_count = message.extra_metadata.get("review_count", 0)
            print(f"\n[Reviewer Agent] Reviewing code (Attempt {review_count + 1}/3)...")
            print(f" └── Code Payload:\n{message.payload}")

            # Safety check to prevent infinite loops (> 3 iterations)
            if review_count >= 3:
                print("[Reviewer Agent] Max review attempts (3) reached. Forcing approval.")
                if human_approve(message.payload):
                    execute_code(message.payload)
                else:
                    print("[Reviewer Agent] Human rejected the code execution. Halting further actions.")
                    return
                success_broadcast = PipelineMessage(
                    payload="SUCCESS: Code approved after reaching maximum allowed review iterations.",
                    message_type="SUCCESS_BROADCAST",
                    sender_id="ReviewerAgent",
                    receiver_id="ALL",
                    trace_id=message.trace_id,
                    current_stage="code_review",
                    target_stage="complete",
                    priority=message.priority
                )
                await self.publish_message(success_broadcast, topic_id=DefaultTopicId())
                return

            # Use LLM to review the code
            messages = [
                SystemMessage(
                    content=(
                        "You are an expert lead code reviewer and security auditor. "
                        "Review the provided Python Flask code for correctness, missing endpoints, bugs, or security flaws. "
                        "If the code is fully correct, safe, and meets standard expectations, reply with exactly the word: 'APPROVED'. "
                        "If there are errors or missing features, start your response with 'REJECTED:' followed by clear, actionable feedback for the developer to fix and include also the full original code you're reviewing."
                    )
                ),
                UserMessage(content=f"Please review this implementation code:\n{message.payload}", source=message.sender_id)
            ]

            llm_response = await self.model_client.create(
                messages=messages,
                cancellation_token=ctx.cancellation_token
            )
            
            review_result = llm_response.content.strip()
            print(f"[Reviewer Agent] LLM Evaluation Result:\n{review_result}")

            if "APPROVED" in review_result.upper() and "REJECTED" not in review_result.upper():
                if human_approve(message.payload):
                    execute_code(message.payload)
                else:
                    print("[Reviewer Agent] Human rejected the code execution. Halting further actions.")
                    return
                print("[Reviewer Agent] Code passed review successfully! Broadcasting success...")
                success_broadcast = PipelineMessage(
                    payload="SUCCESS: The microservice code has passed review!",
                    message_type="SUCCESS_BROADCAST",
                    sender_id="ReviewerAgent",
                    receiver_id="ALL",
                    trace_id=message.trace_id,
                    current_stage="code_review",
                    target_stage="complete",
                    priority=message.priority
                )
                await self.publish_message(success_broadcast, topic_id=DefaultTopicId())
            else:
                print(f"[Reviewer Agent] Code rejected. Sending feedback to CoderAgent (Attempt {review_count + 1}/3)...")
                feedback_msg = PipelineMessage(
                    payload=review_result,
                    message_type="CODE_FEEDBACK",
                    sender_id="ReviewerAgent",
                    receiver_id="CoderAgent",
                    trace_id=message.trace_id,
                    current_stage="code_review",
                    target_stage="coding",
                    priority=message.priority,
                    extra_metadata={"review_count": review_count + 1}
                )
                coder_id = AgentId("CoderAgent", "default")
                await self.send_message(feedback_msg, coder_id)

        elif message.message_type == "ACKNOWLEDGMENT":
            print(f"\n[Reviewer Agent] Received ACK from [{message.sender_id}]: {message.payload}")

async def simulate_conversation(runtime, payload: str = "Build a lightweight REST API service in Flask for a simple In-Memory Key-Value store with TTL."):
    architect_id = AgentId("ArchitectAgent", "default")
        
    initial_message = PipelineMessage(
        payload=payload,
        message_type="FEATURE_REQUEST",
        sender_id="ClientAgent",
        receiver_id="ArchitectAgent",
        trace_id="tr-alpha-9921-prod",
        current_stage="user_prompt",
        target_stage="architecture",
        priority=5,
        extra_metadata={"client": "internal_qa"}
    )

    print("--- Starting AutoGen Core 0.7.5 Pipeline with Iterative Review Loop ---")
    await runtime.send_message(initial_message, architect_id)

def human_approve(code): 
    print(f"Code proposed: {code}") 
    decision = input("Approve this code to run? (y/n): ") 
    return decision.strip().lower() == "y"

def execute_code(code):
    # For demonstration, just print. Never use eval/exec with untrusted code!
    print("Code executed: ", code)

# ==========================================
# 3. RUNTIME INITIALIZATION & EXECUTION
# ==========================================
async def main():
    runtime = SingleThreadedAgentRuntime()

    await ClientAgent.register(
        runtime,
        type="ClientAgent",
        factory=lambda: ClientAgent("Project Client / Initiator")
    )
    await SystemArchitectAgent.register(
        runtime,
        type="ArchitectAgent",
        factory=lambda: SystemArchitectAgent("Senior Software Architect", llm_config=DEFAULT_LLM_CONFIG)
    )
    await BackendCoderAgent.register(
        runtime,
        type="CoderAgent",
        factory=lambda: BackendCoderAgent("Backend Implementation Engineer", llm_config=DEFAULT_LLM_CONFIG)
    )
    await CodeReviewerAgent.register(
        runtime,
        type="ReviewerAgent",
        factory=lambda: CodeReviewerAgent("Lead Security Reviewer", llm_config=DEFAULT_LLM_CONFIG)
    )

    runtime.start()

    feature_request = input("Enter feature request: ")

    await simulate_conversation(runtime, feature_request)

    # Give ample time for back-and-forth iterations, async messages, broadcasts, and ACKs
    await asyncio.sleep(100)
    await runtime.stop()

if __name__ == "__main__":
    asyncio.run(main())