#!/usr/bin/env python3
"""Example script for the MediaAgent."""

from media_utils import MediaAgent


def interactive_mode():
    """Run the agent interactively."""
    print("=" * 50)
    print("Media Agent - AI Image Generation")
    print("=" * 50)
    print()
    print("Examples:")
    print("  - 'Generate a sunset over mountains'")
    print("  - 'Create a cyberpunk city at night'")
    print("  - 'Draw a cat sitting on a beach'")
    print()
    print("Type 'quit' to exit")
    print()

    agent = MediaAgent()

    try:
        while True:
            query = input("\nYou: ").strip()

            if not query:
                continue

            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            print("\nAgent: ", end="", flush=True)
            result = agent.run(query)
            print(result)

    finally:
        agent.unload()


def example_single():
    """Generate a single image."""
    print("Generating image...")

    with MediaAgent() as agent:
        result = agent.run("Generate a photorealistic mountain landscape at sunset with a lake reflection")
        print(result)


def example_direct():
    """Generate using direct API (bypassing LLM)."""
    print("Generating image directly...")

    with MediaAgent() as agent:
        path = agent.generate(
            prompt="A futuristic city with flying cars and neon lights",
            negative_prompt="blurry, low quality",
            resolution="1344x768",
            seed=42,
        )
        print(f"Image saved to: {path}")


def example_batch():
    """Generate multiple images."""
    print("Generating multiple images...")

    prompts = [
        "A serene Japanese garden with cherry blossoms",
        "A dramatic thunderstorm over the ocean",
        "A cozy cabin in a snowy forest",
    ]

    with MediaAgent() as agent:
        for prompt in prompts:
            print(f"\nGenerating: {prompt[:40]}...")
            path = agent.generate(prompt)
            print(f"Saved to: {path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "single":
            example_single()
        elif cmd == "direct":
            example_direct()
        elif cmd == "batch":
            example_batch()
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python test_agent.py [single|direct|batch]")
            print("       python test_agent.py  # interactive mode")
    else:
        interactive_mode()
