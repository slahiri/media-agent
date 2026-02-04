"""MediaAgent CLI entry point."""

import argparse
import sys


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="media-agent",
        description="MediaAgent - AI-powered image generation CLI",
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./outputs",
        help="Output directory for generated images (default: ./outputs)",
    )

    parser.add_argument(
        "-t", "--theme",
        type=str,
        choices=["dark", "light"],
        default="dark",
        help="Color theme (default: dark)",
    )

    parser.add_argument(
        "--no-tips",
        action="store_true",
        help="Don't show getting started tips",
    )

    parser.add_argument(
        "-v", "--version",
        action="store_true",
        help="Show version and exit",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate an image directly")
    gen_parser.add_argument("prompt", type=str, help="Image prompt")
    gen_parser.add_argument("-n", "--negative", type=str, default="", help="Negative prompt")
    gen_parser.add_argument("-r", "--resolution", type=str, default="1024x1024", help="Resolution")
    gen_parser.add_argument("-s", "--seed", type=int, help="Random seed")

    # Download command
    dl_parser = subparsers.add_parser("download", help="Download models")
    dl_parser.add_argument(
        "target",
        choices=["all", "image", "llm"],
        default="all",
        help="What to download",
    )
    dl_parser.add_argument(
        "--mode",
        choices=["pipeline", "split"],
        default="pipeline",
        help="Image model mode",
    )
    dl_parser.add_argument(
        "--local",
        action="store_true",
        help="Copy to local models folder",
    )

    # Settings command
    settings_parser = subparsers.add_parser("settings", help="Manage settings")
    settings_parser.add_argument("action", nargs="?", default="list", help="Action (list, set, get)")
    settings_parser.add_argument("key", nargs="?", help="Setting key")
    settings_parser.add_argument("value", nargs="?", help="Setting value")

    args = parser.parse_args()

    # Handle version
    if args.version:
        from media_agent.cli.components.header import VERSION
        print(f"MediaAgent v{VERSION}")
        return 0

    # Handle subcommands
    if args.command == "generate":
        return _handle_generate(args)

    if args.command == "download":
        return _handle_download(args)

    if args.command == "settings":
        return _handle_settings(args)

    # Run interactive CLI
    try:
        from media_agent.cli.app import run_cli
        run_cli(
            output_dir=args.output,
            theme=args.theme,
            show_tips=not args.no_tips,
        )
        return 0
    except KeyboardInterrupt:
        print("\nGoodbye!")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _handle_generate(args) -> int:
    """Handle direct generation command."""
    from pathlib import Path
    from datetime import datetime
    import uuid

    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from media_agent.image.generator import ImageGenerator
    from media_agent.llm import get_settings

    console = Console()

    try:
        # Check image provider
        settings = get_settings()
        provider = settings.get("image.provider", "local")

        if provider == "nanobanana":
            from media_agent.image.nanobanana import NanobananaGenerator
            api_key = settings.get("nanobanana.api_key")
            if not api_key:
                console.print("[red]Nanobanana API key not set. Use: media-agent settings set nanobanana.api_key <key>[/red]")
                return 1
            generator = NanobananaGenerator(api_key=api_key, output_dir=args.output if hasattr(args, 'output') else "./outputs")
        else:
            mode = settings.get("image.mode", "pipeline")
            generator = ImageGenerator(mode=mode)

        output_dir = Path(args.output if hasattr(args, 'output') else "./outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating image...", total=None)

            image = generator.generate(
                prompt=args.prompt,
                negative_prompt=args.negative if args.negative else None,
                resolution=args.resolution,
                seed=args.seed,
            )

            progress.update(task, description="Saving...")

            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:8]
            filename = f"image_{timestamp}_{unique_id}.png"
            filepath = output_dir / filename
            image.save(filepath)

        console.print(f"[green]✓ Image saved to: {filepath}[/green]")
        return 0

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


def _handle_download(args) -> int:
    """Handle model download command."""
    from media_agent.tools.downloader import (
        download_image_model,
        download_llm_model,
        download_models,
    )

    try:
        if args.target == "all":
            download_models(
                image_mode=args.mode,
                copy_to_local=args.local,
            )
        elif args.target == "image":
            download_image_model(
                mode=args.mode,
                copy_to_local=args.local,
            )
        elif args.target == "llm":
            download_llm_model(copy_to_local=args.local)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _handle_settings(args) -> int:
    """Handle settings command."""
    from media_agent.llm import get_settings

    settings = get_settings()

    if args.action == "list" or (args.action is None and args.key is None):
        # List all settings
        all_settings = settings.all_settings()
        if not all_settings:
            print("No settings configured")
            return 0

        for key, value in _flatten_dict(all_settings):
            # Mask API keys
            if "api_key" in key.lower() and value:
                value = value[:8] + "..." if len(str(value)) > 8 else "***"
            print(f"{key}: {value}")
        return 0

    if args.action == "get":
        if not args.key:
            print("Usage: media-agent settings get <key>", file=sys.stderr)
            return 1
        value = settings.get(args.key)
        if value is None:
            print(f"Setting not found: {args.key}")
            return 1
        # Mask API keys
        if "api_key" in args.key.lower() and value:
            value = value[:8] + "..." if len(str(value)) > 8 else "***"
        print(f"{args.key}: {value}")
        return 0

    if args.action == "set":
        if not args.key or not args.value:
            print("Usage: media-agent settings set <key> <value>", file=sys.stderr)
            return 1
        settings.set(args.key, args.value)
        print(f"Set {args.key}")
        return 0

    # If action is actually a key
    if args.key is None:
        # Get setting
        value = settings.get(args.action)
        if value is None:
            print(f"Setting not found: {args.action}")
            return 1
        if "api_key" in args.action.lower() and value:
            value = value[:8] + "..." if len(str(value)) > 8 else "***"
        print(f"{args.action}: {value}")
        return 0
    else:
        # Set setting
        settings.set(args.action, args.key)
        print(f"Set {args.action}")
        return 0


def _flatten_dict(d: dict, prefix: str = "") -> list:
    """Flatten a nested dict into key-value pairs."""
    items = []
    for key, value in d.items():
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            items.extend(_flatten_dict(value, f"{full_key}."))
        else:
            items.append((full_key, value))
    return items


if __name__ == "__main__":
    sys.exit(main())
